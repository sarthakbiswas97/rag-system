from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from rag.models.document import Chunk, ChunkMetadata
from rag.retrieval.vector_store import VectorStore

VECTOR_SIZE = 4


@pytest.fixture()
def client() -> QdrantClient:
    return QdrantClient(":memory:")


@pytest.fixture()
def store(client: QdrantClient) -> VectorStore:
    vs = VectorStore(client=client, collection="test_collection")
    vs.create_collection(vector_size=VECTOR_SIZE)
    return vs


def _make_chunk(
    text: str,
    embedding: tuple[float, ...],
    doc_id: str = "doc-1",
    source: str = "test.txt",
    tenant_id: str = "",
) -> Chunk:
    return Chunk(
        document_id=doc_id,
        text=text,
        metadata=ChunkMetadata(
            source_file=source,
            tenant_id=tenant_id,
            chunk_index=0,
            total_chunks=1,
        ),
        embedding=embedding,
    )


class TestCreateCollection:
    def test_creates_collection(self, client: QdrantClient) -> None:
        store = VectorStore(client=client, collection="new_col")
        assert not store.collection_exists()
        store.create_collection(vector_size=VECTOR_SIZE)
        assert store.collection_exists()

    def test_skips_if_exists(self, store: VectorStore) -> None:
        # should not raise
        store.create_collection(vector_size=VECTOR_SIZE)
        assert store.collection_exists()


class TestDeleteCollection:
    def test_deletes_collection(self, store: VectorStore) -> None:
        assert store.collection_exists()
        store.delete_collection()
        assert not store.collection_exists()


class TestUpsertChunks:
    def test_upserts_and_counts(self, store: VectorStore) -> None:
        chunks = [
            _make_chunk("hello", (1.0, 0.0, 0.0, 0.0)),
            _make_chunk("world", (0.0, 1.0, 0.0, 0.0)),
        ]
        count = store.upsert_chunks(chunks)
        assert count == 2
        assert store.count() == 2

    def test_empty_list_returns_zero(self, store: VectorStore) -> None:
        assert store.upsert_chunks([]) == 0

    def test_raises_on_missing_embedding(self, store: VectorStore) -> None:
        chunk = Chunk(
            document_id="doc-1",
            text="no embedding",
            metadata=ChunkMetadata(source_file="test.txt"),
        )
        with pytest.raises(ValueError, match="no embedding"):
            store.upsert_chunks([chunk])

    def test_batch_upsert_larger_than_batch_size(self, store: VectorStore) -> None:
        chunks = [
            _make_chunk(f"chunk-{i}", (float(i), 0.0, 0.0, 0.0)) for i in range(15)
        ]
        count = store.upsert_chunks(chunks, batch_size=4)
        assert count == 15
        assert store.count() == 15


class TestSearch:
    def test_finds_exact_match(self, store: VectorStore) -> None:
        target = _make_chunk("target", (1.0, 0.0, 0.0, 0.0))
        other = _make_chunk("other", (0.0, 1.0, 0.0, 0.0))
        store.upsert_chunks([target, other])

        results = store.search(query_embedding=(1.0, 0.0, 0.0, 0.0), top_k=2)
        assert len(results) == 2
        assert results[0].chunk.text == "target"
        assert results[0].score > results[1].score

    def test_returns_scored_chunks(self, store: VectorStore) -> None:
        chunk = _make_chunk("hello", (1.0, 0.0, 0.0, 0.0))
        store.upsert_chunks([chunk])

        results = store.search(query_embedding=(1.0, 0.0, 0.0, 0.0), top_k=1)
        assert len(results) == 1
        assert results[0].retrieval_method == "vector"
        assert results[0].score > 0.9

    def test_empty_collection_returns_empty(self, store: VectorStore) -> None:
        results = store.search(query_embedding=(1.0, 0.0, 0.0, 0.0), top_k=5)
        assert results == ()

    def test_reconstructs_chunk_metadata(self, store: VectorStore) -> None:
        chunk = _make_chunk(
            "metadata test",
            (0.5, 0.5, 0.0, 0.0),
            doc_id="doc-42",
            source="report.pdf",
        )
        store.upsert_chunks([chunk])

        results = store.search(query_embedding=(0.5, 0.5, 0.0, 0.0), top_k=1)
        hit = results[0].chunk
        assert hit.document_id == "doc-42"
        assert hit.text == "metadata test"
        assert hit.metadata.source_file == "report.pdf"
        assert hit.metadata.chunk_index == 0
        assert hit.metadata.total_chunks == 1

    def test_top_k_limits_results(self, store: VectorStore) -> None:
        chunks = [
            _make_chunk(f"chunk-{i}", (float(i % 2), float(i % 3), 0.0, 0.0))
            for i in range(10)
        ]
        store.upsert_chunks(chunks)

        results = store.search(query_embedding=(1.0, 0.0, 0.0, 0.0), top_k=3)
        assert len(results) == 3


class TestTenantIsolation:
    def test_search_filters_by_tenant(self, store: VectorStore) -> None:
        chunk_a = _make_chunk("tenant-a doc", (1.0, 0.0, 0.0, 0.0), tenant_id="t-a")
        chunk_b = _make_chunk("tenant-b doc", (1.0, 0.0, 0.0, 0.0), tenant_id="t-b")
        store.upsert_chunks([chunk_a, chunk_b])

        results = store.search(
            query_embedding=(1.0, 0.0, 0.0, 0.0), top_k=10, tenant_id="t-a"
        )
        assert len(results) == 1
        assert results[0].chunk.metadata.tenant_id == "t-a"

    def test_search_without_tenant_returns_all(self, store: VectorStore) -> None:
        chunk_a = _make_chunk("a", (1.0, 0.0, 0.0, 0.0), tenant_id="t-a")
        chunk_b = _make_chunk("b", (0.9, 0.1, 0.0, 0.0), tenant_id="t-b")
        store.upsert_chunks([chunk_a, chunk_b])

        results = store.search(query_embedding=(1.0, 0.0, 0.0, 0.0), top_k=10)
        assert len(results) == 2

    def test_count_by_tenant(self, store: VectorStore) -> None:
        chunks = [
            _make_chunk("a1", (1.0, 0.0, 0.0, 0.0), tenant_id="t-a"),
            _make_chunk("a2", (0.0, 1.0, 0.0, 0.0), tenant_id="t-a"),
            _make_chunk("b1", (0.0, 0.0, 1.0, 0.0), tenant_id="t-b"),
        ]
        store.upsert_chunks(chunks)

        assert store.count_by_tenant("t-a") == 2
        assert store.count_by_tenant("t-b") == 1
        assert store.count_by_tenant("t-none") == 0

    def test_delete_by_tenant(self, store: VectorStore) -> None:
        chunks = [
            _make_chunk("a1", (1.0, 0.0, 0.0, 0.0), tenant_id="t-a"),
            _make_chunk("b1", (0.0, 1.0, 0.0, 0.0), tenant_id="t-b"),
        ]
        store.upsert_chunks(chunks)

        store.delete_by_tenant("t-a")
        assert store.count_by_tenant("t-a") == 0
        assert store.count_by_tenant("t-b") == 1

    def test_tenant_id_persists_in_metadata(self, store: VectorStore) -> None:
        chunk = _make_chunk("test", (1.0, 0.0, 0.0, 0.0), tenant_id="t-x")
        store.upsert_chunks([chunk])

        results = store.search(
            query_embedding=(1.0, 0.0, 0.0, 0.0), top_k=1, tenant_id="t-x"
        )
        assert results[0].chunk.metadata.tenant_id == "t-x"
