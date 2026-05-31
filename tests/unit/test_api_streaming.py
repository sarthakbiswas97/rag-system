from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag.api.dependencies import (
    get_db_session,
    get_llm_client,
    get_retriever,
    get_session_store,
    get_verification_pipeline,
)
from rag.api.routes_query import router as query_router
from rag.models.document import Chunk, ChunkMetadata
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
        query="What is the capital?",
        rewritten_query=None,
        scored_chunks=(
            ScoredChunk(chunk=chunk, score=0.95, retrieval_method="vector"),
        ),
        retrieval_time_ms=12.5,
    )


@pytest.fixture()
def mock_tenant() -> Tenant:
    t = MagicMock(spec=Tenant)
    t.id = "test-tenant-id"
    t.name = "test"
    t.email = "test@example.com"
    t.status = TenantStatus.ACTIVE
    return t


@pytest.fixture()
def mock_db_session() -> AsyncMock:
    return AsyncMock()


@pytest.fixture()
def client(mock_tenant: Tenant, mock_db_session: AsyncMock) -> TestClient:
    app = FastAPI()
    app.include_router(query_router)

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = _make_retrieval_result()

    mock_llm = MagicMock()

    async def _mock_stream(*args, **kwargs):
        yield "Paris"
        yield " is"
        yield " the"
        yield " capital."
        from rag.models.generation import LLMResponse

        yield LLMResponse(
            content="Paris is the capital.",
            prompt_tokens=10,
            completion_tokens=4,
            model="gpt-4o-mini",
            elapsed_ms=50.0,
        )

    mock_llm.stream_generate = _mock_stream

    app.dependency_overrides[get_retriever] = lambda: mock_retriever
    app.dependency_overrides[get_llm_client] = lambda: mock_llm
    app.dependency_overrides[get_verification_pipeline] = lambda: None
    app.dependency_overrides[get_session_store] = lambda: None
    app.dependency_overrides[get_db_session] = lambda: mock_db_session
    app.dependency_overrides[get_current_tenant] = lambda: mock_tenant

    return TestClient(app)


class TestQueryStreamEndpoint:
    def test_stream_returns_sse(self, client: TestClient) -> None:
        with patch(
            "rag.api.routes_query.rewrite_with_context",
            return_value="What is the capital?",
        ):
            response = client.post(
                "/v1/query/stream",
                json={"question": "What is the capital of France?"},
            )

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        events = []
        for line in response.text.strip().split("\n\n"):
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

        token_events = [e for e in events if e.get("type") == "token"]
        assert len(token_events) == 4
        assert token_events[0]["content"] == "Paris"

        done_events = [e for e in events if e.get("type") == "done"]
        assert len(done_events) == 1
        assert "answer" in done_events[0]

    def test_stream_abstention_when_no_results(
        self, mock_tenant: Tenant, mock_db_session: AsyncMock
    ) -> None:
        app = FastAPI()
        app.include_router(query_router)

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = RetrievalResult(
            query="xyz",
            rewritten_query=None,
            scored_chunks=(),
            retrieval_time_ms=5.0,
        )

        async def _empty_stream(*args, **kwargs):
            from rag.models.generation import LLMResponse

            yield LLMResponse(
                content="",
                prompt_tokens=0,
                completion_tokens=0,
                model="gpt-4o-mini",
                elapsed_ms=0.0,
            )

        mock_llm = MagicMock()
        mock_llm.stream_generate = _empty_stream

        app.dependency_overrides[get_retriever] = lambda: mock_retriever
        app.dependency_overrides[get_llm_client] = lambda: mock_llm
        app.dependency_overrides[get_verification_pipeline] = lambda: None
        app.dependency_overrides[get_session_store] = lambda: None
        app.dependency_overrides[get_db_session] = lambda: mock_db_session
        app.dependency_overrides[get_current_tenant] = lambda: mock_tenant

        client = TestClient(app)

        with patch("rag.api.routes_query.rewrite_with_context", return_value="xyz"):
            response = client.post(
                "/v1/query/stream",
                json={"question": "xyz nonsense"},
            )

        assert response.status_code == 200

        events = []
        for line in response.text.strip().split("\n\n"):
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

        abstention_events = [e for e in events if e.get("type") == "abstention"]
        assert len(abstention_events) == 1
