from __future__ import annotations

from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag.api.dependencies import (
    get_db_session,
    get_pipeline,
    get_query_cache,
    get_vector_store,
)
from rag.api.routes_documents import router as documents_router
from rag.models.ingestion import DocumentInfo, IngestionResult
from rag.tenancy.auth import get_current_tenant
from rag.tenancy.document_repository import DocumentRepository
from rag.tenancy.models import Document, DocumentStatus, Tenant, TenantStatus


@pytest.fixture()
def mock_vector_store() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def mock_pipeline() -> MagicMock:
    pipeline = MagicMock()
    pipeline.ingest_documents.return_value = IngestionResult(
        documents_processed=1,
        documents_skipped=0,
        documents_failed=0,
        chunks_created=5,
        elapsed_ms=100.0,
        processed_documents=(
            DocumentInfo(
                document_id="doc-new",
                source_file="updated.txt",
                content_hash="abc123",
                chunk_count=5,
            ),
        ),
    )
    return pipeline


@pytest.fixture()
def mock_db_session() -> AsyncMock:
    return AsyncMock()


@pytest.fixture()
def mock_tenant() -> Tenant:
    tenant = MagicMock(spec=Tenant)
    tenant.id = "test-tenant-id"
    tenant.name = "test"
    tenant.email = "test@example.com"
    tenant.status = TenantStatus.ACTIVE
    return tenant


@pytest.fixture()
def client(
    mock_vector_store: MagicMock,
    mock_pipeline: MagicMock,
    mock_db_session: AsyncMock,
    mock_tenant: Tenant,
) -> TestClient:
    app = FastAPI()
    app.include_router(documents_router)

    app.dependency_overrides[get_vector_store] = lambda: mock_vector_store
    app.dependency_overrides[get_pipeline] = lambda: mock_pipeline
    app.dependency_overrides[get_db_session] = lambda: mock_db_session
    app.dependency_overrides[get_query_cache] = lambda: None
    app.dependency_overrides[get_current_tenant] = lambda: mock_tenant

    return TestClient(app)


def _make_doc(doc_id: str, source_file: str, chunk_count: int = 1) -> Document:
    doc = MagicMock(spec=Document)
    doc.id = doc_id
    doc.source_file = source_file
    doc.chunk_count = chunk_count
    doc.status = DocumentStatus.ACTIVE
    doc.created_at.isoformat.return_value = "2024-01-01T00:00:00"
    doc.updated_at.isoformat.return_value = "2024-01-01T00:00:00"
    return doc


class TestListDocuments:
    def test_list_documents_returns_200(self, client: TestClient) -> None:
        mock_docs = [_make_doc("d1", "a.txt"), _make_doc("d2", "b.txt")]

        with patch.object(
            DocumentRepository, "list_by_tenant", new=AsyncMock(return_value=mock_docs)
        ):
            resp = client.get("/v1/documents")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["documents"]) == 2
        assert data["total"] == 2

    def test_list_documents_pagination(self, client: TestClient) -> None:
        with patch.object(
            DocumentRepository, "list_by_tenant", new=AsyncMock(return_value=[])
        ):
            resp = client.get("/v1/documents?skip=0&limit=10")

        assert resp.status_code == 200


class TestDeleteDocument:
    def test_delete_document_returns_200(
        self,
        client: TestClient,
        mock_vector_store: MagicMock,
    ) -> None:
        mock_doc = _make_doc("doc-1", "a.txt")

        with (
            patch.object(
                DocumentRepository,
                "get_by_id",
                new=AsyncMock(return_value=mock_doc),
            ),
            patch.object(
                DocumentRepository,
                "soft_delete",
                new=AsyncMock(return_value=mock_doc),
            ),
        ):
            resp = client.delete("/v1/documents/doc-1")

        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted"] is True
        assert data["document_id"] == "doc-1"
        mock_vector_store.delete_by_document_id.assert_called_once_with(
            "doc-1", tenant_id="test-tenant-id"
        )

    def test_delete_document_404_when_not_found(self, client: TestClient) -> None:
        with patch.object(
            DocumentRepository,
            "get_by_id",
            new=AsyncMock(return_value=None),
        ):
            resp = client.delete("/v1/documents/missing")

        assert resp.status_code == 404


class TestUpdateDocument:
    def test_update_document_returns_200(
        self,
        client: TestClient,
        mock_vector_store: MagicMock,
        mock_pipeline: MagicMock,
    ) -> None:
        mock_doc = _make_doc("doc-1", "old.txt")

        with (
            patch.object(
                DocumentRepository,
                "get_by_id",
                new=AsyncMock(return_value=mock_doc),
            ),
            patch.object(
                DocumentRepository,
                "update_status",
                new=AsyncMock(return_value=mock_doc),
            ),
            patch.object(
                DocumentRepository,
                "update_document",
                new=AsyncMock(return_value=mock_doc),
            ),
        ):
            resp = client.put(
                "/v1/documents/doc-1",
                files={"file": ("updated.txt", BytesIO(b"new content"), "text/plain")},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["documents_processed"] == 1
        mock_vector_store.delete_by_document_id.assert_called_once_with(
            "doc-1", tenant_id="test-tenant-id"
        )
        mock_pipeline.ingest_documents.assert_called_once()

    def test_update_document_404_when_not_found(self, client: TestClient) -> None:
        with patch.object(
            DocumentRepository,
            "get_by_id",
            new=AsyncMock(return_value=None),
        ):
            resp = client.put(
                "/v1/documents/missing",
                files={"file": ("file.txt", BytesIO(b"content"), "text/plain")},
            )

        assert resp.status_code == 404
