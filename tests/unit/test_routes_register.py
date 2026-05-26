from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag.api.dependencies import get_db_session
from rag.api.routes_register import router
from rag.tenancy.models import Tenant, TenantStatus


def _make_tenant(name: str = "Acme", email: str = "acme@example.com") -> MagicMock:
    tenant = MagicMock(spec=Tenant)
    tenant.id = "tenant-abc"
    tenant.name = name
    tenant.email = email
    tenant.status = TenantStatus.ACTIVE
    tenant.embedding_model_version = None
    tenant.created_at.isoformat.return_value = "2026-01-01T00:00:00"
    tenant.updated_at.isoformat.return_value = "2026-01-01T00:00:00"
    return tenant


@pytest.fixture()
def mock_db_session() -> AsyncMock:
    return AsyncMock()


@pytest.fixture()
def client(mock_db_session: AsyncMock) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db_session] = lambda: mock_db_session
    return TestClient(app)


class TestRegister:
    def test_register_success(self, client: TestClient) -> None:
        tenant = _make_tenant()
        with patch(
            "rag.api.routes_register.TenantRepository"
        ) as mock_repo:
            instance = mock_repo.return_value
            instance.create = AsyncMock(return_value=(tenant, "rk_testkey123"))

            resp = client.post(
                "/v1/register",
                json={"name": "Acme", "email": "acme@example.com"},
            )

        assert resp.status_code == 201
        data = resp.json()
        assert data["tenant"]["id"] == "tenant-abc"
        assert data["tenant"]["name"] == "Acme"
        assert data["tenant"]["email"] == "acme@example.com"
        assert data["api_key"] == "rk_testkey123"

    def test_duplicate_returns_409(self, client: TestClient) -> None:
        from sqlalchemy.exc import IntegrityError

        with patch(
            "rag.api.routes_register.TenantRepository"
        ) as mock_repo:
            instance = mock_repo.return_value
            instance.create = AsyncMock(
                side_effect=IntegrityError("dup", {}, Exception())
            )

            resp = client.post(
                "/v1/register",
                json={"name": "Acme", "email": "acme@example.com"},
            )

        assert resp.status_code == 409
        assert "already registered" in resp.json()["detail"]

    def test_missing_email_returns_422(self, client: TestClient) -> None:
        resp = client.post("/v1/register", json={"name": "Acme"})
        assert resp.status_code == 422

    def test_invalid_email_returns_422(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/register", json={"name": "Acme", "email": "not-an-email"}
        )
        assert resp.status_code == 422

    def test_blank_name_returns_422(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/register", json={"name": "   ", "email": "a@b.com"}
        )
        assert resp.status_code == 422

    def test_missing_name_returns_422(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/register", json={"email": "a@b.com"}
        )
        assert resp.status_code == 422
