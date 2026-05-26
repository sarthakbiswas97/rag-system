from __future__ import annotations

from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag.api.dependencies import (
    get_db_session,
    get_generator,
    get_job_store,
    get_llm_client,
    get_pipeline,
    get_retriever,
    get_session_store,
    get_vector_store,
    get_verification_pipeline,
)
from rag.api.routes_health import router as health_router
from rag.api.routes_ingest import router as ingest_router
from rag.api.routes_query import router as query_router
from rag.models.document import Chunk, ChunkMetadata
from rag.models.generation import Citation, GenerationResponse
from rag.models.ingestion import IngestionResult
from rag.models.retrieval import RetrievalResult, ScoredChunk
from rag.tenancy.auth import get_current_tenant
from rag.tenancy.models import Tenant, TenantStatus


def _make_retrieval_result() -> RetrievalResult:
    chunk = Chunk(
        chunk_id="chunk-abc",
        document_id="doc-1",
        text="Paris is the capital of France.",
        metadata=ChunkMetadata(source_file="geo.pdf"),
    )
    return RetrievalResult(
        query="What is the capital of France?",
        rewritten_query=None,
        scored_chunks=(
            ScoredChunk(chunk=chunk, score=0.95, retrieval_method="vector"),
        ),
        retrieval_time_ms=12.5,
    )


def _make_generation_response(retrieval: RetrievalResult) -> GenerationResponse:
    return GenerationResponse(
        answer="Paris is the capital of France [1].",
        citations=(
            Citation(
                chunk_id="chunk-abc",
                source_file="geo.pdf",
                text_snippet="Paris is the capital of France.",
                sentence_index=0,
            ),
        ),
        is_abstention=False,
        confidence_score=0.95,
        retrieval_result=retrieval,
        generation_time_ms=500.0,
    )


@pytest.fixture()
def mock_retriever() -> MagicMock:
    retriever = MagicMock()
    retriever.retrieve.return_value = _make_retrieval_result()
    return retriever


@pytest.fixture()
def mock_generator() -> MagicMock:
    generator = MagicMock()

    async def _generate(question, retrieval, top_k=5):
        return _make_generation_response(retrieval)

    generator.generate = _generate
    return generator


@pytest.fixture()
def mock_pipeline() -> MagicMock:
    pipeline = MagicMock()
    pipeline.ingest_documents.return_value = IngestionResult(
        documents_processed=1,
        documents_skipped=0,
        documents_failed=0,
        chunks_created=3,
        elapsed_ms=100.0,
    )
    return pipeline


@pytest.fixture()
def mock_vector_store() -> MagicMock:
    store = MagicMock()
    store.collection_exists.return_value = True
    return store


@pytest.fixture()
def mock_db_session() -> AsyncMock:
    session = AsyncMock()
    session.execute.return_value = None
    return session


@pytest.fixture()
def mock_tenant() -> Tenant:
    return Tenant(
        id="test-tenant-id",
        name="test",
        email="test@example.com",
        api_key_hash="fake",
        status=TenantStatus.ACTIVE,
    )


@pytest.fixture()
def client(
    mock_retriever: MagicMock,
    mock_generator: MagicMock,
    mock_pipeline: MagicMock,
    mock_vector_store: MagicMock,
    mock_db_session: AsyncMock,
    mock_tenant: Tenant,
) -> TestClient:
    app = FastAPI()
    app.include_router(query_router)
    app.include_router(ingest_router)
    app.include_router(health_router)

    mock_tenant = MagicMock(spec=Tenant)
    mock_tenant.id = "test-tenant-id"
    mock_tenant.name = "test"
    mock_tenant.email = "test@example.com"
    mock_tenant.status = TenantStatus.ACTIVE

    app.dependency_overrides[get_retriever] = lambda: mock_retriever
    app.dependency_overrides[get_generator] = lambda: mock_generator
    app.dependency_overrides[get_pipeline] = lambda: mock_pipeline
    app.dependency_overrides[get_vector_store] = lambda: mock_vector_store
    app.dependency_overrides[get_verification_pipeline] = lambda: None
    app.dependency_overrides[get_llm_client] = lambda: MagicMock()
    app.dependency_overrides[get_job_store] = lambda: MagicMock()
    app.dependency_overrides[get_session_store] = lambda: None
    app.dependency_overrides[get_db_session] = lambda: mock_db_session
    app.dependency_overrides[get_current_tenant] = lambda: mock_tenant

    return TestClient(app)


class TestQueryEndpoint:
    def test_valid_query_returns_200(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/query",
            json={"question": "What is the capital of France?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "Paris is the capital of France [1]."
        assert len(data["citations"]) == 1
        assert data["citations"][0]["source"] == "geo.pdf"
        assert data["is_abstention"] is False
        assert data["confidence"] == 0.95
        assert "timing" in data
        assert data["timing"]["retrieval_ms"] >= 0
        assert data["timing"]["generation_ms"] >= 0
        assert data["timing"]["total_ms"] >= 0

    def test_empty_question_returns_422(self, client: TestClient) -> None:
        resp = client.post("/v1/query", json={"question": ""})
        assert resp.status_code == 422

    def test_blank_question_returns_422(self, client: TestClient) -> None:
        resp = client.post("/v1/query", json={"question": "   "})
        assert resp.status_code == 422

    def test_missing_question_returns_422(self, client: TestClient) -> None:
        resp = client.post("/v1/query", json={})
        assert resp.status_code == 422

    def test_custom_top_k(self, client: TestClient, mock_retriever: MagicMock) -> None:
        client.post(
            "/v1/query",
            json={"question": "test", "top_k": 10},
        )
        mock_retriever.retrieve.assert_called_once_with(
            "test", top_k=10, tenant_id="test-tenant-id"
        )

    def test_verification_null_when_disabled(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/query",
            json={"question": "What is the capital of France?"},
        )
        data = resp.json()
        assert data["verification"] is None

    def test_top_k_out_of_range_returns_422(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/query",
            json={"question": "test", "top_k": 0},
        )
        assert resp.status_code == 422


class TestIngestEndpoint:
    def test_upload_file_returns_200(self, client: TestClient) -> None:
        file_content = b"Hello world test content."
        resp = client.post(
            "/v1/ingest",
            files=[("files", ("test.txt", BytesIO(file_content), "text/plain"))],
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["documents_processed"] == 1
        assert data["chunks_created"] == 3

    def test_no_files_returns_422(self, client: TestClient) -> None:
        resp = client.post("/v1/ingest")
        assert resp.status_code == 422


class TestHealthEndpoint:
    def test_healthy(self, client: TestClient) -> None:
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["qdrant"] == "connected"
        assert data["database"] == "connected"

    def test_degraded_when_collection_missing(
        self, client: TestClient, mock_vector_store: MagicMock
    ) -> None:
        mock_vector_store.collection_exists.return_value = False
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["qdrant"] == "unreachable"

    def test_degraded_on_qdrant_exception(
        self, client: TestClient, mock_vector_store: MagicMock
    ) -> None:
        mock_vector_store.collection_exists.side_effect = Exception(
            "connection refused"
        )
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"

    def test_degraded_on_db_failure(
        self, client: TestClient, mock_db_session: AsyncMock
    ) -> None:
        mock_db_session.execute.side_effect = Exception("db unreachable")
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["database"] == "unreachable"
