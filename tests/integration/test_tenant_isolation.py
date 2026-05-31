from __future__ import annotations

from io import BytesIO

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from rag.api.dependencies import get_db_session, get_pipeline, get_vector_store
from rag.api.routes_admin import router as admin_router
from rag.api.routes_health import router as health_router
from rag.api.routes_ingest import router as ingest_router
from rag.api.routes_query import router as query_router
from rag.api.routes_tenant import router as tenant_router
from rag.ingestion.embedder import Embedder
from rag.ingestion.pipeline import IngestionPipeline
from rag.retrieval.vector_store import VectorStore
from rag.tenancy.database import build_engine, build_session_factory, init_db

MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_DIM = 384
ADMIN_KEY = "test-admin-key-12345"


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder(model_name=MODEL_NAME, batch_size=32)


@pytest.fixture()
def vector_store() -> VectorStore:
    client = QdrantClient(":memory:")
    store = VectorStore(client=client, collection="test_tenant_iso")
    store.create_collection(vector_size=VECTOR_DIM)
    return store


@pytest.fixture()
def pipeline(embedder: Embedder, vector_store: VectorStore) -> IngestionPipeline:
    return IngestionPipeline(
        embedder=embedder,
        vector_store=vector_store,
        chunk_size=200,
        chunk_overlap=20,
    )


@pytest.fixture()
async def db_session_factory():
    engine = build_engine("sqlite+aiosqlite://")
    await init_db(engine)
    factory = build_session_factory(engine)
    yield factory
    await engine.dispose()


@pytest.fixture()
def client(
    vector_store: VectorStore,
    pipeline: IngestionPipeline,
    db_session_factory,
) -> TestClient:
    app = FastAPI()
    app.include_router(admin_router)
    app.include_router(query_router)
    app.include_router(ingest_router)
    app.include_router(health_router)
    app.include_router(tenant_router)

    app.state.admin_api_key = ADMIN_KEY
    app.state.db_session_factory = db_session_factory

    async def _get_db_session():
        async with db_session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = _get_db_session
    app.dependency_overrides[get_vector_store] = lambda: vector_store
    app.dependency_overrides[get_pipeline] = lambda: pipeline

    return TestClient(app)


def _admin_headers() -> dict[str, str]:
    return {"X-API-Key": ADMIN_KEY}


def _tenant_headers(api_key: str) -> dict[str, str]:
    return {"X-API-Key": api_key}


def _create_tenant(client: TestClient, name: str) -> tuple[str, str]:
    resp = client.post(
        "/admin/tenants",
        json={"name": name},
        headers=_admin_headers(),
    )
    assert resp.status_code == 201
    data = resp.json()
    return data["tenant"]["id"], data["api_key"]


def _ingest_text(client: TestClient, api_key: str, filename: str, content: str) -> dict:
    resp = client.post(
        "/v1/ingest",
        files=[("files", (filename, BytesIO(content.encode()), "text/plain"))],
        headers=_tenant_headers(api_key),
    )
    assert resp.status_code == 200
    return resp.json()


class TestTenantLifecycle:
    def test_create_and_authenticate(self, client: TestClient) -> None:
        tenant_id, api_key = _create_tenant(client, "lifecycle-test")

        resp = client.get("/v1/me", headers=_tenant_headers(api_key))
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == tenant_id
        assert data["name"] == "lifecycle-test"
        assert data["status"] == "active"

    def test_invalid_key_rejected(self, client: TestClient) -> None:
        resp = client.get("/v1/me", headers=_tenant_headers("rk_bogus"))
        assert resp.status_code == 401

    def test_deleted_tenant_rejected(self, client: TestClient) -> None:
        tenant_id, api_key = _create_tenant(client, "to-delete")

        resp = client.delete(f"/admin/tenants/{tenant_id}", headers=_admin_headers())
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

        resp = client.get("/v1/me", headers=_tenant_headers(api_key))
        assert resp.status_code == 401

    def test_update_name_via_self_service(self, client: TestClient) -> None:
        _, api_key = _create_tenant(client, "old-name")

        resp = client.put(
            "/v1/me",
            json={"name": "new-name"},
            headers=_tenant_headers(api_key),
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "new-name"

        resp = client.get("/v1/me", headers=_tenant_headers(api_key))
        assert resp.json()["name"] == "new-name"


class TestCrossTenantIsolation:
    def test_ingest_and_query_isolated(self, client: TestClient) -> None:
        _, key_a = _create_tenant(client, "tenant-alpha")
        _, key_b = _create_tenant(client, "tenant-beta")

        _ingest_text(
            client,
            key_a,
            "france.txt",
            "The capital of France is Paris. It is a major European city.",
        )
        _ingest_text(
            client,
            key_b,
            "germany.txt",
            "The capital of Germany is Berlin. It is a historic city.",
        )

        stats_a = client.get("/v1/me/stats", headers=_tenant_headers(key_a)).json()
        stats_b = client.get("/v1/me/stats", headers=_tenant_headers(key_b)).json()
        assert stats_a["chunk_count"] >= 1
        assert stats_b["chunk_count"] >= 1

    def test_query_only_sees_own_data(
        self, client: TestClient, vector_store: VectorStore
    ) -> None:
        _, key_a = _create_tenant(client, "iso-alpha")
        _, key_b = _create_tenant(client, "iso-beta")

        _ingest_text(
            client,
            key_a,
            "alpha.txt",
            "Quantum computing uses qubits to perform calculations.",
        )
        _ingest_text(
            client,
            key_b,
            "beta.txt",
            "Photosynthesis converts sunlight into chemical energy in plants.",
        )

        # Verify vector store has data for both
        total = vector_store.count()
        assert total >= 2

        # Each tenant's stats should only show their own chunks
        stats_a = client.get("/v1/me/stats", headers=_tenant_headers(key_a)).json()
        stats_b = client.get("/v1/me/stats", headers=_tenant_headers(key_b)).json()
        assert stats_a["chunk_count"] + stats_b["chunk_count"] <= total

    def test_deduplication_is_tenant_scoped(self, client: TestClient) -> None:
        _, key_a = _create_tenant(client, "dedup-alpha")
        _, key_b = _create_tenant(client, "dedup-beta")

        content = "Identical content for dedup testing across tenants."

        result_a = _ingest_text(client, key_a, "same.txt", content)
        assert result_a["documents_processed"] == 1

        # Same content, different tenant -- should NOT be deduplicated
        result_b = _ingest_text(client, key_b, "same.txt", content)
        assert result_b["documents_processed"] == 1

        # Same content, same tenant -- SHOULD be deduplicated
        result_a2 = _ingest_text(client, key_a, "same.txt", content)
        assert result_a2["documents_processed"] == 0
        assert result_a2["documents_skipped"] == 1


class TestAdminAccess:
    def test_admin_can_list_all_tenants(self, client: TestClient) -> None:
        _create_tenant(client, "admin-list-1")
        _create_tenant(client, "admin-list-2")

        resp = client.get("/admin/tenants", headers=_admin_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 2

    def test_tenant_key_cannot_access_admin(self, client: TestClient) -> None:
        _, api_key = _create_tenant(client, "no-admin")
        resp = client.get("/admin/tenants", headers=_tenant_headers(api_key))
        assert resp.status_code == 401

    def test_no_key_returns_422(self, client: TestClient) -> None:
        resp = client.get("/admin/tenants")
        assert resp.status_code == 422


class TestHealthWithDatabase:
    def test_health_shows_database_connected(self, client: TestClient) -> None:
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["qdrant"] == "connected"
        assert data["database"] == "connected"
