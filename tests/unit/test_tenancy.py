from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from rag.tenancy.database import build_engine, build_session_factory, init_db
from rag.tenancy.models import (
    FineTuneJob,
    FineTuneStatus,
    ModelType,
    Tenant,
    TenantStatus,
)
from rag.tenancy.repository import (
    API_KEY_PREFIX,
    TenantRepository,
    generate_api_key,
    hash_api_key,
)

DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture()
async def db_session() -> AsyncSession:
    engine = build_engine(DB_URL)
    await init_db(engine)
    session_factory = build_session_factory(engine)
    async with session_factory() as session:
        yield session
    await engine.dispose()


@pytest.fixture()
async def repo(db_session: AsyncSession) -> TenantRepository:
    return TenantRepository(db_session)


class TestApiKeyUtils:
    def test_generate_api_key_has_prefix(self) -> None:
        key = generate_api_key()
        assert key.startswith(API_KEY_PREFIX)

    def test_generate_api_key_is_unique(self) -> None:
        keys = {generate_api_key() for _ in range(100)}
        assert len(keys) == 100

    def test_generate_api_key_length(self) -> None:
        key = generate_api_key()
        assert len(key) == len(API_KEY_PREFIX) + 64  # 32 bytes = 64 hex chars

    def test_hash_api_key_deterministic(self) -> None:
        key = "rk_abc123"
        assert hash_api_key(key) == hash_api_key(key)

    def test_hash_api_key_different_for_different_keys(self) -> None:
        assert hash_api_key("rk_aaa") != hash_api_key("rk_bbb")


class TestTenantRepository:
    async def test_create_tenant(self, repo: TenantRepository) -> None:
        tenant, api_key = await repo.create("acme")
        assert tenant.name == "acme"
        assert tenant.status == TenantStatus.ACTIVE
        assert tenant.id is not None
        assert api_key.startswith(API_KEY_PREFIX)
        assert tenant.embedding_model_version is None

    async def test_get_by_id(self, repo: TenantRepository) -> None:
        tenant, _ = await repo.create("acme")
        found = await repo.get_by_id(tenant.id)
        assert found is not None
        assert found.name == "acme"

    async def test_get_by_id_not_found(self, repo: TenantRepository) -> None:
        result = await repo.get_by_id("nonexistent-id")
        assert result is None

    async def test_get_by_api_key(self, repo: TenantRepository) -> None:
        tenant, api_key = await repo.create("acme")
        found = await repo.get_by_api_key(api_key)
        assert found is not None
        assert found.id == tenant.id

    async def test_get_by_api_key_invalid(self, repo: TenantRepository) -> None:
        await repo.create("acme")
        result = await repo.get_by_api_key("rk_invalid_key")
        assert result is None

    async def test_get_by_api_key_excludes_deleted(
        self, repo: TenantRepository
    ) -> None:
        tenant, api_key = await repo.create("acme")
        await repo.soft_delete(tenant.id)
        result = await repo.get_by_api_key(api_key)
        assert result is None

    async def test_list_all(self, repo: TenantRepository) -> None:
        await repo.create("tenant-a")
        await repo.create("tenant-b")
        tenants = await repo.list_all()
        assert len(tenants) == 2

    async def test_list_all_excludes_deleted(self, repo: TenantRepository) -> None:
        t1, _ = await repo.create("tenant-a")
        await repo.create("tenant-b")
        await repo.soft_delete(t1.id)
        tenants = await repo.list_all()
        assert len(tenants) == 1
        assert tenants[0].name == "tenant-b"

    async def test_list_all_includes_deleted(self, repo: TenantRepository) -> None:
        t1, _ = await repo.create("tenant-a")
        await repo.create("tenant-b")
        await repo.soft_delete(t1.id)
        tenants = await repo.list_all(include_deleted=True)
        assert len(tenants) == 2

    async def test_soft_delete(self, repo: TenantRepository) -> None:
        tenant, _ = await repo.create("acme")
        deleted = await repo.soft_delete(tenant.id)
        assert deleted is not None
        assert deleted.status == TenantStatus.DELETED

    async def test_soft_delete_nonexistent(self, repo: TenantRepository) -> None:
        result = await repo.soft_delete("nonexistent-id")
        assert result is None

    async def test_update_status(self, repo: TenantRepository) -> None:
        tenant, _ = await repo.create("acme")
        updated = await repo.update_status(tenant.id, TenantStatus.SUSPENDED)
        assert updated is not None
        assert updated.status == TenantStatus.SUSPENDED

    async def test_update_name(self, repo: TenantRepository) -> None:
        tenant, _ = await repo.create("old-name")
        updated = await repo.update_name(tenant.id, "new-name")
        assert updated is not None
        assert updated.name == "new-name"

    async def test_update_name_nonexistent(self, repo: TenantRepository) -> None:
        result = await repo.update_name("nonexistent-id", "name")
        assert result is None

    async def test_created_at_is_set(self, repo: TenantRepository) -> None:
        tenant, _ = await repo.create("acme")
        assert tenant.created_at is not None

    async def test_duplicate_name_raises(self, repo: TenantRepository) -> None:
        await repo.create("acme")
        with pytest.raises(Exception):
            await repo.create("acme")


class TestTenantModel:
    def test_tenant_status_values(self) -> None:
        assert TenantStatus.ACTIVE.value == "active"
        assert TenantStatus.SUSPENDED.value == "suspended"
        assert TenantStatus.DELETED.value == "deleted"

    def test_fine_tune_status_values(self) -> None:
        assert FineTuneStatus.PENDING.value == "pending"
        assert FineTuneStatus.RUNNING.value == "running"
        assert FineTuneStatus.COMPLETED.value == "completed"
        assert FineTuneStatus.FAILED.value == "failed"

    def test_model_type_values(self) -> None:
        assert ModelType.EMBEDDING.value == "embedding"
        assert ModelType.LLM.value == "llm"


class TestDatabase:
    async def test_init_creates_tables(self) -> None:
        engine = build_engine(DB_URL)
        await init_db(engine)
        session_factory = build_session_factory(engine)
        async with session_factory() as session:
            repo = TenantRepository(session)
            tenants = await repo.list_all()
            assert tenants == []
        await engine.dispose()
