from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from rag.api.dependencies import (
    get_db_session,
    get_pipeline,
    get_query_cache,
    get_vector_store,
)
from rag.api.schemas import DocumentListResponse, DocumentOut, IngestResponse
from rag.ingestion.pipeline import IngestionPipeline
from rag.retrieval.cache import QueryCache
from rag.retrieval.vector_store import VectorStore
from rag.tenancy.auth import get_current_tenant
from rag.tenancy.document_repository import DocumentRepository
from rag.tenancy.models import DocumentStatus, EventType, Tenant
from rag.tenancy.usage import UsageTracker

logger = logging.getLogger(__name__)

router = APIRouter()


def _doc_to_out(doc) -> DocumentOut:
    return DocumentOut(
        id=doc.id,
        source_file=doc.source_file,
        chunk_count=doc.chunk_count,
        status=doc.status.value,
        created_at=doc.created_at.isoformat() if doc.created_at else "",
        updated_at=doc.updated_at.isoformat() if doc.updated_at else "",
    )


@router.get("/v1/documents", response_model=DocumentListResponse)
async def list_documents(
    tenant: Tenant = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
    skip: int = 0,
    limit: int = 100,
) -> DocumentListResponse:
    repo = DocumentRepository(session)
    docs = await repo.list_by_tenant(tenant.id, skip=skip, limit=limit)
    return DocumentListResponse(
        documents=[_doc_to_out(d) for d in docs],
        total=len(docs),
    )


@router.delete("/v1/documents/{document_id}")
async def delete_document(
    document_id: str,
    tenant: Tenant = Depends(get_current_tenant),
    vector_store: VectorStore = Depends(get_vector_store),
    query_cache: QueryCache | None = Depends(get_query_cache),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    repo = DocumentRepository(session)
    doc = await repo.get_by_id(document_id, tenant.id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete chunks from Qdrant
    vector_store.delete_by_document_id(document_id, tenant_id=tenant.id)

    # Soft-delete in DB
    await repo.soft_delete(document_id, tenant.id)

    # Invalidate cache
    if query_cache is not None:
        query_cache.invalidate_tenant(tenant.id)

    logger.info(
        "Document deleted",
        extra={"document_id": document_id, "tenant_id": tenant.id},
    )

    return {"deleted": True, "document_id": document_id}


@router.put("/v1/documents/{document_id}", response_model=IngestResponse)
async def update_document(
    document_id: str,
    file: UploadFile,
    tenant: Tenant = Depends(get_current_tenant),
    pipeline: IngestionPipeline = Depends(get_pipeline),
    vector_store: VectorStore = Depends(get_vector_store),
    query_cache: QueryCache | None = Depends(get_query_cache),
    session: AsyncSession = Depends(get_db_session),
) -> IngestResponse:
    repo = DocumentRepository(session)
    doc = await repo.get_by_id(document_id, tenant.id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    # Mark as updating
    await repo.update_status(document_id, tenant.id, DocumentStatus.UPDATING)

    # Delete old chunks
    vector_store.delete_by_document_id(document_id, tenant_id=tenant.id)

    # Save uploaded file to temp
    tmp_dir = Path(tempfile.mkdtemp())
    dest = tmp_dir / (file.filename or "unnamed")
    content = await file.read()
    dest.write_bytes(content)

    try:
        # Re-ingest
        result = pipeline.ingest_documents([dest], tenant_id=tenant.id)

        if result.documents_processed == 0:
            await repo.update_status(document_id, tenant.id, DocumentStatus.ACTIVE)
            raise HTTPException(status_code=400, detail="Document update failed")

        # Update document record with new info
        info = result.processed_documents[0]
        await repo.update_document(
            document_id,
            tenant.id,
            source_file=info.source_file,
            chunk_count=info.chunk_count,
            content_hash=info.content_hash,
            status=DocumentStatus.ACTIVE,
        )

        if query_cache is not None and result.chunks_created > 0:
            query_cache.invalidate_tenant(tenant.id)

        tracker = UsageTracker(session)
        await tracker.record(
            tenant_id=tenant.id,
            event_type=EventType.INGEST,
            value=result.chunks_created,
            elapsed_ms=result.elapsed_ms,
        )

        return IngestResponse(
            documents_processed=result.documents_processed,
            documents_skipped=result.documents_skipped,
            documents_failed=result.documents_failed,
            chunks_created=result.chunks_created,
            elapsed_ms=result.elapsed_ms,
        )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
