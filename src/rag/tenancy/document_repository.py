from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag.models.ingestion import DocumentInfo
from rag.tenancy.models import Document, DocumentStatus

logger = logging.getLogger(__name__)


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        doc_id: str,
        tenant_id: str,
        source_file: str,
        chunk_count: int,
        content_hash: str,
    ) -> Document:
        doc = Document(
            id=doc_id,
            tenant_id=tenant_id,
            source_file=source_file,
            chunk_count=chunk_count,
            content_hash=content_hash,
            status=DocumentStatus.ACTIVE,
        )
        self._session.add(doc)
        await self._session.commit()
        await self._session.refresh(doc)
        logger.info(
            "Document record created",
            extra={"document_id": doc_id, "tenant_id": tenant_id},
        )
        return doc

    async def create_many(
        self,
        infos: list[DocumentInfo],
        tenant_id: str,
    ) -> None:
        """Batch-insert document records in a single transaction."""
        if not infos:
            return

        docs = [
            Document(
                id=info.document_id,
                tenant_id=tenant_id,
                source_file=info.source_file,
                chunk_count=info.chunk_count,
                content_hash=info.content_hash,
                status=DocumentStatus.ACTIVE,
            )
            for info in infos
        ]
        self._session.add_all(docs)
        await self._session.commit()
        logger.info(
            "Document records batch created",
            extra={"count": len(docs), "tenant_id": tenant_id},
        )

    async def get_by_id(self, doc_id: str, tenant_id: str) -> Document | None:
        stmt = select(Document).where(
            Document.id == doc_id,
            Document.tenant_id == tenant_id,
            Document.status != DocumentStatus.DELETED,
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_by_tenant(
        self,
        tenant_id: str,
        *,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Document]:
        stmt = (
            select(Document)
            .where(
                Document.tenant_id == tenant_id,
                Document.status != DocumentStatus.DELETED,
            )
            .order_by(Document.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def update_status(
        self, doc_id: str, tenant_id: str, status: DocumentStatus
    ) -> Document | None:
        doc = await self.get_by_id(doc_id, tenant_id)
        if doc is None:
            return None
        doc.status = status
        await self._session.commit()
        await self._session.refresh(doc)
        logger.info(
            "Document status updated",
            extra={"document_id": doc_id, "status": status.value},
        )
        return doc

    async def soft_delete(self, doc_id: str, tenant_id: str) -> Document | None:
        return await self.update_status(doc_id, tenant_id, DocumentStatus.DELETED)

    async def update_document(
        self,
        doc_id: str,
        tenant_id: str,
        *,
        source_file: str | None = None,
        chunk_count: int | None = None,
        content_hash: str | None = None,
        status: DocumentStatus | None = None,
    ) -> Document | None:
        doc = await self.get_by_id(doc_id, tenant_id)
        if doc is None:
            return None
        if source_file is not None:
            doc.source_file = source_file
        if chunk_count is not None:
            doc.chunk_count = chunk_count
        if content_hash is not None:
            doc.content_hash = content_hash
        if status is not None:
            doc.status = status
        await self._session.commit()
        await self._session.refresh(doc)
        logger.info(
            "Document updated",
            extra={"document_id": doc_id, "tenant_id": tenant_id},
        )
        return doc
