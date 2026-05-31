from __future__ import annotations

import uuid

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

from rag.models.document import Chunk, ChunkMetadata
from rag.retrieval.vector_store import VectorStore

VECTOR_DIM = 384


@pytest.fixture()
def store() -> VectorStore:
    client = QdrantClient(":memory:")
    store = VectorStore(client=client, collection="test_sparse_store")
    store.create_collection(vector_size=VECTOR_DIM)
    return store


class TestVectorStoreSparseVectors:
    def test_collection_has_sparse_config(self, store: VectorStore) -> None:
        info = store._client.get_collection(store._collection)
        assert info.config.params.sparse_vectors is not None
        assert "text" in info.config.params.sparse_vectors

    def test_upsert_sparse_vectors(self, store: VectorStore) -> None:
        chunk_id = str(uuid.uuid4())
        chunks = [
            Chunk(
                chunk_id=chunk_id,
                document_id="doc-1",
                text="hello world",
                metadata=ChunkMetadata(source_file="test.txt"),
                embedding=tuple(float(i) for i in range(VECTOR_DIM)),
            ),
        ]
        store.upsert_chunks(chunks)
        store.upsert_sparse_vectors(
            [chunk_id],
            [SparseVector(indices=[0, 1], values=[1.0, 0.5])],
        )

        # Sparse search should find the point
        results = store.search_sparse(SparseVector(indices=[0], values=[1.0]), top_k=5)
        assert len(results) == 1
        assert results[0].chunk.chunk_id == chunk_id

    def test_search_sparse_with_tenant_filter(self, store: VectorStore) -> None:
        cid_a = str(uuid.uuid4())
        cid_b = str(uuid.uuid4())
        chunks_a = [
            Chunk(
                chunk_id=cid_a,
                document_id="doc-a",
                text="tenant a content",
                metadata=ChunkMetadata(source_file="a.txt", tenant_id="tenant-a"),
                embedding=tuple(float(i) for i in range(VECTOR_DIM)),
            ),
        ]
        chunks_b = [
            Chunk(
                chunk_id=cid_b,
                document_id="doc-b",
                text="tenant b content",
                metadata=ChunkMetadata(source_file="b.txt", tenant_id="tenant-b"),
                embedding=tuple(float(i) for i in range(VECTOR_DIM)),
            ),
        ]
        store.upsert_chunks(chunks_a + chunks_b)
        store.upsert_sparse_vectors(
            [cid_a, cid_b],
            [
                SparseVector(indices=[0], values=[1.0]),
                SparseVector(indices=[0], values=[1.0]),
            ],
        )

        results = store.search_sparse(
            SparseVector(indices=[0], values=[1.0]),
            top_k=10,
            tenant_id="tenant-a",
        )
        assert len(results) == 1
        assert results[0].chunk.metadata.tenant_id == "tenant-a"

    def test_search_sparse_empty_store(self, store: VectorStore) -> None:
        results = store.search_sparse(SparseVector(indices=[0], values=[1.0]), top_k=5)
        assert results == ()

    def test_upsert_sparse_empty_list(self, store: VectorStore) -> None:
        count = store.upsert_sparse_vectors([], [])
        assert count == 0

    def test_delete_by_document_id(self, store: VectorStore) -> None:
        chunks = [
            Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id="doc-1",
                text="content",
                metadata=ChunkMetadata(source_file="test.txt", tenant_id="t1"),
                embedding=tuple(float(i) for i in range(VECTOR_DIM)),
            ),
            Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id="doc-2",
                text="other",
                metadata=ChunkMetadata(source_file="test.txt", tenant_id="t1"),
                embedding=tuple(float(i) for i in range(VECTOR_DIM)),
            ),
        ]
        store.upsert_chunks(chunks)
        assert store.count() == 2

        store.delete_by_document_id("doc-1", tenant_id="t1")
        assert store.count() == 1

        # Verify only doc-2 remains
        results = store.search(
            tuple(float(i) for i in range(VECTOR_DIM)), top_k=10, tenant_id="t1"
        )
        assert len(results) == 1
        assert results[0].chunk.document_id == "doc-2"
