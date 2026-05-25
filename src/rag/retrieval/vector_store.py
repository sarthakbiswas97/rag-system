from __future__ import annotations

import logging
from collections.abc import Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from rag.models.document import Chunk, ChunkMetadata
from rag.models.retrieval import ScoredChunk

logger = logging.getLogger(__name__)


def _chunk_to_payload(chunk: Chunk) -> dict:
    return {
        "text": chunk.text,
        "document_id": chunk.document_id,
        "tenant_id": chunk.metadata.tenant_id,
        "source_file": chunk.metadata.source_file,
        "page_number": chunk.metadata.page_number,
        "section_title": chunk.metadata.section_title,
        "chunk_index": chunk.metadata.chunk_index,
        "total_chunks": chunk.metadata.total_chunks,
        "parent_chunk_id": chunk.metadata.parent_chunk_id,
        "created_at": chunk.metadata.created_at,
    }


def _payload_to_chunk(point_id: str, payload: dict) -> Chunk:
    return Chunk(
        chunk_id=point_id,
        document_id=payload.get("document_id", ""),
        text=payload.get("text", ""),
        metadata=ChunkMetadata(
            source_file=payload.get("source_file", ""),
            tenant_id=payload.get("tenant_id", ""),
            page_number=payload.get("page_number"),
            section_title=payload.get("section_title"),
            chunk_index=payload.get("chunk_index", 0),
            total_chunks=payload.get("total_chunks", 0),
            parent_chunk_id=payload.get("parent_chunk_id"),
            created_at=payload.get("created_at", ""),
        ),
    )


class VectorStore:
    def __init__(self, client: QdrantClient, collection: str) -> None:
        self._client = client
        self._collection = collection

    def collection_exists(self) -> bool:
        return self._client.collection_exists(self._collection)

    def create_collection(self, vector_size: int) -> None:
        if self.collection_exists():
            logger.info(
                "Collection already exists, skipping creation",
                extra={"collection": self._collection},
            )
            return

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.DOT,
            ),
        )

        # create payload indexes before ingestion (Qdrant best practice)
        self._client.create_payload_index(
            collection_name=self._collection,
            field_name="document_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        self._client.create_payload_index(
            collection_name=self._collection,
            field_name="source_file",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        self._client.create_payload_index(
            collection_name=self._collection,
            field_name="tenant_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )

        logger.info(
            "Created collection",
            extra={
                "collection": self._collection,
                "vector_size": vector_size,
                "distance": "Dot",
            },
        )

    def delete_collection(self) -> None:
        self._client.delete_collection(self._collection)
        logger.info(
            "Deleted collection",
            extra={"collection": self._collection},
        )

    def count(self) -> int:
        info = self._client.get_collection(self._collection)
        return info.points_count

    def upsert_chunks(self, chunks: Sequence[Chunk], batch_size: int = 100) -> int:
        if not chunks:
            return 0

        points = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(
                    f"Chunk {chunk.chunk_id} has no embedding. "
                    "Embed chunks before upserting."
                )
            points.append(
                PointStruct(
                    id=chunk.chunk_id,
                    vector=list(chunk.embedding),
                    payload=_chunk_to_payload(chunk),
                )
            )

        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self._client.upsert(
                collection_name=self._collection,
                points=batch,
                wait=True,
            )

        logger.info(
            "Upserted chunks",
            extra={
                "collection": self._collection,
                "count": len(points),
                "batches": (len(points) + batch_size - 1) // batch_size,
            },
        )

        return len(points)

    def search(
        self,
        query_embedding: tuple[float, ...],
        top_k: int = 50,
        tenant_id: str = "",
    ) -> tuple[ScoredChunk, ...]:
        query_filter = None
        if tenant_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="tenant_id", match=MatchValue(value=tenant_id)
                    )
                ]
            )

        results = self._client.query_points(
            collection_name=self._collection,
            query=list(query_embedding),
            query_filter=query_filter,
            limit=top_k,
        ).points

        scored_chunks = tuple(
            ScoredChunk(
                chunk=_payload_to_chunk(str(hit.id), hit.payload or {}),
                score=hit.score,
                retrieval_method="vector",
            )
            for hit in results
        )

        return scored_chunks

    def delete_by_tenant(self, tenant_id: str) -> None:
        self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="tenant_id", match=MatchValue(value=tenant_id)
                    )
                ]
            ),
        )
        logger.info(
            "Deleted tenant chunks",
            extra={"collection": self._collection, "tenant_id": tenant_id},
        )

    def count_by_tenant(self, tenant_id: str) -> int:
        result = self._client.count(
            collection_name=self._collection,
            count_filter=Filter(
                must=[
                    FieldCondition(
                        key="tenant_id", match=MatchValue(value=tenant_id)
                    )
                ]
            ),
            exact=True,
        )
        return result.count
