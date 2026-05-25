from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from rag.api.routes_admin import router
from rag.tenancy.database import build_engine, build_session_factory, init_db

DB_URL = "sqlite+aiosqlite:///:memory:"
ADMIN_KEY = "test-admin-key"


@pytest.fixture()
async def app():
    engine = build_engine(DB_URL)
    await init_db(engine)
    session_factory = build_session_factory(engine)

    app = FastAPI()
    app.state.db_session_factory = session_factory
    app.state.admin_api_key = ADMIN_KEY
    app.include_router(router)

    yield app
    await engine.dispose()


@pytest.fixture()
async def client(app: FastAPI) -> AsyncClient:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


HEADERS = {"X-API-Key": ADMIN_KEY}


class TestCreateTenant:
    async def test_create_tenant(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/admin/tenants", json={"name": "acme"}, headers=HEADERS
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["tenant"]["name"] == "acme"
        assert data["tenant"]["status"] == "active"
        assert data["api_key"].startswith("rk_")

    async def test_create_duplicate_name(self, client: AsyncClient) -> None:
        await client.post(
            "/admin/tenants", json={"name": "acme"}, headers=HEADERS
        )
        resp = await client.post(
            "/admin/tenants", json={"name": "acme"}, headers=HEADERS
        )
        assert resp.status_code == 409

    async def test_create_blank_name(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/admin/tenants", json={"name": "  "}, headers=HEADERS
        )
        assert resp.status_code == 422

    async def test_create_without_admin_key(self, client: AsyncClient) -> None:
        resp = await client.post("/admin/tenants", json={"name": "acme"})
        assert resp.status_code == 422

    async def test_create_with_wrong_admin_key(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/admin/tenants",
            json={"name": "acme"},
            headers={"X-API-Key": "wrong"},
        )
        assert resp.status_code == 401


class TestListTenants:
    async def test_list_empty(self, client: AsyncClient) -> None:
        resp = await client.get("/admin/tenants", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["tenants"] == []
        assert data["count"] == 0

    async def test_list_after_create(self, client: AsyncClient) -> None:
        await client.post(
            "/admin/tenants", json={"name": "a"}, headers=HEADERS
        )
        await client.post(
            "/admin/tenants", json={"name": "b"}, headers=HEADERS
        )
        resp = await client.get("/admin/tenants", headers=HEADERS)
        assert resp.json()["count"] == 2

    async def test_list_excludes_deleted(self, client: AsyncClient) -> None:
        create_resp = await client.post(
            "/admin/tenants", json={"name": "doomed"}, headers=HEADERS
        )
        tenant_id = create_resp.json()["tenant"]["id"]
        await client.delete(f"/admin/tenants/{tenant_id}", headers=HEADERS)

        resp = await client.get("/admin/tenants", headers=HEADERS)
        assert resp.json()["count"] == 0


class TestGetTenant:
    async def test_get_tenant(self, client: AsyncClient) -> None:
        create_resp = await client.post(
            "/admin/tenants", json={"name": "acme"}, headers=HEADERS
        )
        tenant_id = create_resp.json()["tenant"]["id"]
        resp = await client.get(f"/admin/tenants/{tenant_id}", headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["name"] == "acme"

    async def test_get_not_found(self, client: AsyncClient) -> None:
        resp = await client.get("/admin/tenants/nonexistent", headers=HEADERS)
        assert resp.status_code == 404


class TestDeleteTenant:
    async def test_soft_delete(self, client: AsyncClient) -> None:
        create_resp = await client.post(
            "/admin/tenants", json={"name": "acme"}, headers=HEADERS
        )
        tenant_id = create_resp.json()["tenant"]["id"]
        resp = await client.delete(f"/admin/tenants/{tenant_id}", headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

    async def test_delete_not_found(self, client: AsyncClient) -> None:
        resp = await client.delete("/admin/tenants/nonexistent", headers=HEADERS)
        assert resp.status_code == 404

    async def test_deleted_tenant_still_visible_by_id(
        self, client: AsyncClient
    ) -> None:
        create_resp = await client.post(
            "/admin/tenants", json={"name": "acme"}, headers=HEADERS
        )
        tenant_id = create_resp.json()["tenant"]["id"]
        await client.delete(f"/admin/tenants/{tenant_id}", headers=HEADERS)

        resp = await client.get(f"/admin/tenants/{tenant_id}", headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"
