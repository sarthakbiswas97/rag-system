from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from rag.generation.generator import Generator
from rag.generation.llm_client import LLMClient
from rag.ingestion.job import JobStore
from rag.ingestion.pipeline import IngestionPipeline
from rag.retrieval.retriever import Retriever
from rag.retrieval.vector_store import VectorStore
from rag.session.store import SessionStore
from rag.verification.pipeline import VerificationPipeline


def get_retriever(request: Request) -> Retriever:
    return request.app.state.retriever


def get_generator(request: Request) -> Generator:
    return request.app.state.generator


def get_pipeline(request: Request) -> IngestionPipeline:
    return request.app.state.pipeline


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store


def get_verification_pipeline(request: Request) -> VerificationPipeline | None:
    return getattr(request.app.state, "verification_pipeline", None)


def get_job_store(request: Request) -> JobStore:
    return request.app.state.job_store


def get_llm_client(request: Request) -> LLMClient:
    return request.app.state.llm_client


def get_session_store(request: Request) -> SessionStore | None:
    return getattr(request.app.state, "session_store", None)


async def get_db_session(request: Request) -> AsyncIterator[AsyncSession]:
    session_factory = request.app.state.db_session_factory
    async with session_factory() as session:
        yield session
