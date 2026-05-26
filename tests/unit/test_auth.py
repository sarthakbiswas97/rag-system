from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from httpx import ASGITransport, AsyncClient

from rag.tenancy.auth import get_current_tenant, require_admin
from rag.tenancy.database import build_engine, build_session_factory, init_db
from rag.tenancy.models import Tenant, TenantStatus
from rag.tenancy.repository import TenantRepository

DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture()
async def db_setup():
    engine = build_engine(DB_URL)
    await init_db(engine)
    session_factory = build_session_factory(engine)
    yield engine, session_factory
    await engine.dispose()


@pytest.fixture()
async def tenant_and_key(db_setup) -> tuple[Tenant, str]:
    _, session_factory = db_setup
    async with session_factory() as session:
        repo = TenantRepository(session)
        tenant, api_key = await repo.create("test-tenant", email="test@test.com")
        return tenant, api_key


def _build_app(db_setup, admin_key: str = "test-admin-key") -> FastAPI:
    _, session_factory = db_setup

    app = FastAPI()
    app.state.db_session_factory = session_factory
    app.state.admin_api_key = admin_key

    @app.get("/protected")
    async def protected(tenant: Tenant = Depends(get_current_tenant)):
        return {"tenant_id": tenant.id, "name": tenant.name}

    @app.get("/admin")
    async def admin(
        _: None = Depends(require_admin),
    ):
        return {"admin": True}

    return app


class TestGetCurrentTenant:
    async def test_valid_api_key(self, db_setup, tenant_and_key) -> None:
        tenant, api_key = tenant_and_key
        app = _build_app(db_setup)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/protected", headers={"X-API-Key": api_key})
        assert resp.status_code == 200
        assert resp.json()["tenant_id"] == tenant.id

    async def test_missing_api_key(self, db_setup) -> None:
        app = _build_app(db_setup)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/protected")
        assert resp.status_code == 422  # FastAPI validation error for missing header

    async def test_invalid_api_key(self, db_setup) -> None:
        app = _build_app(db_setup)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/protected", headers={"X-API-Key": "rk_invalid"})
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Invalid API key"

    async def test_suspended_tenant(self, db_setup, tenant_and_key) -> None:
        tenant, api_key = tenant_and_key
        _, session_factory = db_setup
        async with session_factory() as session:
            repo = TenantRepository(session)
            await repo.update_status(tenant.id, TenantStatus.SUSPENDED)

        app = _build_app(db_setup)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/protected", headers={"X-API-Key": api_key})
        assert resp.status_code == 403
        assert resp.json()["detail"] == "Tenant account is suspended"

    async def test_deleted_tenant(self, db_setup, tenant_and_key) -> None:
        tenant, api_key = tenant_and_key
        _, session_factory = db_setup
        async with session_factory() as session:
            repo = TenantRepository(session)
            await repo.soft_delete(tenant.id)

        app = _build_app(db_setup)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/protected", headers={"X-API-Key": api_key})
        assert resp.status_code == 401


class TestRequireAdmin:
    async def test_valid_admin_key(self, db_setup) -> None:
        app = _build_app(db_setup, admin_key="secret-admin")
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/admin", headers={"X-API-Key": "secret-admin"})
        assert resp.status_code == 200
        assert resp.json()["admin"] is True

    async def test_invalid_admin_key(self, db_setup) -> None:
        app = _build_app(db_setup, admin_key="secret-admin")
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/admin", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 401

    async def test_missing_admin_key_config(self, db_setup) -> None:
        app = _build_app(db_setup, admin_key="")
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/admin", headers={"X-API-Key": "any-key"})
        assert resp.status_code == 503
        assert resp.json()["detail"] == "Admin access not configured"
