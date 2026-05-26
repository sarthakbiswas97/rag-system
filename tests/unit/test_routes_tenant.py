from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag.api.dependencies import get_db_session, get_vector_store
from rag.api.routes_tenant import router
from rag.tenancy.auth import get_current_tenant
from rag.tenancy.models import Tenant, TenantStatus


def _make_mock_tenant() -> MagicMock:
    tenant = MagicMock(spec=Tenant)
    tenant.id = "tenant-123"
    tenant.name = "Acme Corp"
    tenant.email = "acme@example.com"
    tenant.status = TenantStatus.ACTIVE
    tenant.embedding_model_version = None
    tenant.created_at.isoformat.return_value = "2026-01-01T00:00:00"
    tenant.updated_at.isoformat.return_value = "2026-01-01T00:00:00"
    return tenant


@pytest.fixture()
def mock_tenant() -> MagicMock:
    return _make_mock_tenant()


@pytest.fixture()
def mock_vector_store() -> MagicMock:
    store = MagicMock()
    store.count_by_tenant.return_value = 42
    return store


@pytest.fixture()
def mock_db_session() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def client(
    mock_tenant: MagicMock,
    mock_vector_store: MagicMock,
    mock_db_session: MagicMock,
) -> TestClient:
    app = FastAPI()
    app.include_router(router)

    app.dependency_overrides[get_current_tenant] = lambda: mock_tenant
    app.dependency_overrides[get_vector_store] = lambda: mock_vector_store
    app.dependency_overrides[get_db_session] = lambda: mock_db_session

    return TestClient(app)


class TestGetMe:
    def test_returns_tenant_info(self, client: TestClient) -> None:
        resp = client.get("/v1/me")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "tenant-123"
        assert data["name"] == "Acme Corp"
        assert data["status"] == "active"
        assert data["embedding_model_version"] is None

    def test_includes_timestamps(self, client: TestClient) -> None:
        resp = client.get("/v1/me")
        data = resp.json()
        assert "created_at" in data
        assert "updated_at" in data


class TestGetMeStats:
    def test_returns_chunk_count(
        self, client: TestClient, mock_vector_store: MagicMock
    ) -> None:
        resp = client.get("/v1/me/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["tenant_id"] == "tenant-123"
        assert data["chunk_count"] == 42
        mock_vector_store.count_by_tenant.assert_called_once_with("tenant-123")

    def test_zero_chunks(
        self, client: TestClient, mock_vector_store: MagicMock
    ) -> None:
        mock_vector_store.count_by_tenant.return_value = 0
        resp = client.get("/v1/me/stats")
        assert resp.status_code == 200
        assert resp.json()["chunk_count"] == 0


class TestUpdateMe:
    def test_update_name(self, client: TestClient, mock_db_session: MagicMock) -> None:
        updated_tenant = _make_mock_tenant()
        updated_tenant.name = "New Name"
        updated_tenant.updated_at.isoformat.return_value = "2026-01-02T00:00:00"

        # Patch the repository inside the route
        import rag.api.routes_tenant as mod

        original_repo_cls = mod.TenantRepository

        class FakeRepo:
            def __init__(self, session):
                pass

            async def update_name(self, tenant_id, name):
                return updated_tenant

        mod.TenantRepository = FakeRepo  # type: ignore[misc]
        try:
            resp = client.put("/v1/me", json={"name": "New Name"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["name"] == "New Name"
        finally:
            mod.TenantRepository = original_repo_cls  # type: ignore[misc]

    def test_blank_name_returns_422(self, client: TestClient) -> None:
        resp = client.put("/v1/me", json={"name": "   "})
        assert resp.status_code == 422

    def test_empty_name_returns_422(self, client: TestClient) -> None:
        resp = client.put("/v1/me", json={"name": ""})
        assert resp.status_code == 422

    def test_missing_name_returns_422(self, client: TestClient) -> None:
        resp = client.put("/v1/me", json={})
        assert resp.status_code == 422
