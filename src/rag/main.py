from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient
from redis import Redis
from starlette.requests import Request

from rag.api.middleware import RequestContextMiddleware
from rag.api.routes_admin import router as admin_router
from rag.api.routes_health import router as health_router
from rag.api.routes_ingest import router as ingest_router
from rag.api.routes_query import router as query_router
from rag.api.routes_tenant import router as tenant_router
from rag.config import get_settings
from rag.generation.generator import Generator
from rag.generation.llm_client import LLMClient
from rag.ingestion.embedder import Embedder
from rag.ingestion.pipeline import IngestionPipeline
from rag.retrieval.bm25_store import BM25Store
from rag.retrieval.reranker import Reranker
from rag.retrieval.retriever import Retriever
from rag.retrieval.vector_store import VectorStore
from rag.session.store import SessionStore
from rag.tenancy.database import build_engine, build_session_factory, close_db, init_db
from rag.verification.abstention import AbstentionDecider
from rag.verification.citation_validator import CitationValidator
from rag.verification.entailment import EntailmentChecker
from rag.verification.pipeline import VerificationPipeline

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    embedder = Embedder(
        model_name=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )

    client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )
    vector_store = VectorStore(client=client, collection=settings.qdrant_collection)
    vector_store.create_collection(vector_size=embedder.dimension)

    reranker = None
    if settings.enable_reranking:
        reranker = Reranker(model_name=settings.reranker_model)

    bm25_store = BM25Store() if settings.enable_hybrid_search else None

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        reranker=reranker,
        bm25_store=bm25_store,
    )

    llm_client = LLMClient(
        api_key=settings.openai_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout,
    )
    generator = Generator(llm_client=llm_client)

    pipeline = IngestionPipeline(
        embedder=embedder,
        vector_store=vector_store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        bm25_store=bm25_store,
    )

    db_engine = build_engine(settings.database_url)
    await init_db(db_engine)
    db_session_factory = build_session_factory(db_engine)

    verification_pipeline = None
    if settings.enable_verification:
        entailment_checker = EntailmentChecker(model_name=settings.nli_model)
        citation_validator = CitationValidator(
            entailment_checker=entailment_checker,
            support_threshold=settings.citation_support_threshold,
        )
        abstention_decider = AbstentionDecider(
            retrieval_score_threshold=settings.retrieval_score_threshold,
            reranker_score_threshold=settings.reranker_score_threshold,
            faithfulness_threshold=settings.faithfulness_threshold,
        )
        verification_pipeline = VerificationPipeline(
            entailment_checker=entailment_checker,
            citation_validator=citation_validator,
            abstention_decider=abstention_decider,
        )

    redis_client = Redis.from_url(settings.redis_url)
    session_store = SessionStore(client=redis_client)

    app.state.session_store = session_store
    app.state.llm_client = llm_client
    app.state.retriever = retriever
    app.state.generator = generator
    app.state.pipeline = pipeline
    app.state.vector_store = vector_store
    app.state.verification_pipeline = verification_pipeline
    app.state.db_session_factory = db_session_factory
    app.state.admin_api_key = settings.admin_api_key

    logger.info("Application started")
    yield

    redis_client.close()
    await close_db(db_engine)
    logger.info("Application shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG System",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(RequestContextMiddleware)

    app.include_router(query_router)
    app.include_router(ingest_router)
    app.include_router(health_router)
    app.include_router(admin_router)
    app.include_router(tenant_router)

    @app.exception_handler(Exception)
    async def global_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "Unhandled error",
            extra={"path": request.url.path},
        )
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"},
        )

    return app


app = create_app()
