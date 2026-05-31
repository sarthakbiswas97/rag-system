from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from rag.models.ingestion import DocumentInfo
from rag.tenancy.database import build_engine, build_session_factory, init_db
from rag.tenancy.document_repository import DocumentRepository
from rag.tenancy.models import DocumentStatus

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
async def repo(db_session: AsyncSession) -> DocumentRepository:
    return DocumentRepository(db_session)


class TestDocumentRepository:
    async def test_create_document(self, repo: DocumentRepository) -> None:
        doc = await repo.create(
            doc_id="doc-1",
            tenant_id="tenant-a",
            source_file="test.txt",
            chunk_count=5,
            content_hash="abc123",
        )
        assert doc.id == "doc-1"
        assert doc.tenant_id == "tenant-a"
        assert doc.source_file == "test.txt"
        assert doc.chunk_count == 5
        assert doc.status == DocumentStatus.ACTIVE

    async def test_get_by_id(self, repo: DocumentRepository) -> None:
        await repo.create(
            doc_id="doc-1",
            tenant_id="tenant-a",
            source_file="test.txt",
            chunk_count=3,
            content_hash="hash",
        )
        doc = await repo.get_by_id("doc-1", "tenant-a")
        assert doc is not None
        assert doc.source_file == "test.txt"

    async def test_get_by_id_wrong_tenant(self, repo: DocumentRepository) -> None:
        await repo.create(
            doc_id="doc-1",
            tenant_id="tenant-a",
            source_file="test.txt",
            chunk_count=3,
            content_hash="hash",
        )
        doc = await repo.get_by_id("doc-1", "tenant-b")
        assert doc is None

    async def test_list_by_tenant(self, repo: DocumentRepository) -> None:
        await repo.create("d1", "tenant-a", "a.txt", 1, "h1")
        await repo.create("d2", "tenant-a", "b.txt", 2, "h2")
        await repo.create("d3", "tenant-b", "c.txt", 3, "h3")

        docs = await repo.list_by_tenant("tenant-a")
        assert len(docs) == 2
        assert {d.id for d in docs} == {"d1", "d2"}

    async def test_list_pagination(self, repo: DocumentRepository) -> None:
        for i in range(5):
            await repo.create(f"d{i}", "tenant-a", f"{i}.txt", 1, f"h{i}")

        docs = await repo.list_by_tenant("tenant-a", skip=2, limit=2)
        assert len(docs) == 2

    async def test_soft_delete(self, repo: DocumentRepository) -> None:
        await repo.create("d1", "tenant-a", "a.txt", 1, "h1")
        doc = await repo.soft_delete("d1", "tenant-a")
        assert doc is not None
        assert doc.status == DocumentStatus.DELETED

        # Should not appear in list
        docs = await repo.list_by_tenant("tenant-a")
        assert len(docs) == 0

    async def test_update_document(self, repo: DocumentRepository) -> None:
        await repo.create("d1", "tenant-a", "old.txt", 1, "h1")
        doc = await repo.update_document(
            "d1", "tenant-a", source_file="new.txt", chunk_count=10
        )
        assert doc is not None
        assert doc.source_file == "new.txt"
        assert doc.chunk_count == 10

    async def test_update_status(self, repo: DocumentRepository) -> None:
        await repo.create("d1", "tenant-a", "a.txt", 1, "h1")
        doc = await repo.update_status("d1", "tenant-a", DocumentStatus.UPDATING)
        assert doc is not None
        assert doc.status == DocumentStatus.UPDATING

    async def test_create_many_inserts_all(self, repo: DocumentRepository) -> None:
        infos = [
            DocumentInfo(
                document_id="d1", source_file="a.txt", content_hash="h1", chunk_count=3
            ),
            DocumentInfo(
                document_id="d2", source_file="b.txt", content_hash="h2", chunk_count=5
            ),
        ]
        await repo.create_many(infos, tenant_id="tenant-a")

        docs = await repo.list_by_tenant("tenant-a")
        assert len(docs) == 2
        assert {d.id for d in docs} == {"d1", "d2"}

    async def test_create_many_empty_list_is_noop(
        self, repo: DocumentRepository
    ) -> None:
        await repo.create_many([], tenant_id="tenant-a")
        docs = await repo.list_by_tenant("tenant-a")
        assert len(docs) == 0
