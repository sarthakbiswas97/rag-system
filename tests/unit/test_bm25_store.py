from __future__ import annotations

from rag.models.document import Chunk, ChunkMetadata
from rag.retrieval.bm25_store import BM25Store


def _make_chunk(
    chunk_id: str, text: str, tenant_id: str = ""
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        text=text,
        metadata=ChunkMetadata(source_file="test.txt", tenant_id=tenant_id),
    )


class TestBM25Store:
    def test_empty_store_returns_empty(self) -> None:
        store = BM25Store()
        result = store.search("anything", top_k=5)
        assert result == ()

    def test_search_returns_relevant_results(self) -> None:
        store = BM25Store()
        store.add_chunks([
            _make_chunk("c1", "Python is a programming language"),
            _make_chunk("c2", "The capital of France is Paris"),
            _make_chunk("c3", "Machine learning uses Python extensively"),
        ])

        result = store.search("Python programming", top_k=2)

        assert len(result) == 2
        texts = [sc.chunk.text for sc in result]
        assert "Python is a programming language" in texts

    def test_retrieval_method_is_bm25(self) -> None:
        store = BM25Store()
        store.add_chunks([_make_chunk("c1", "test document content")])

        result = store.search("test", top_k=1)

        assert len(result) == 1
        assert result[0].retrieval_method == "bm25"

    def test_top_k_limits_results(self) -> None:
        store = BM25Store()
        chunks = [
            _make_chunk(f"c{i}", f"document about topic {i}")
            for i in range(10)
        ]
        store.add_chunks(chunks)

        result = store.search("document topic", top_k=3)

        assert len(result) == 3

    def test_scores_are_nonzero(self) -> None:
        store = BM25Store()
        store.add_chunks([
            _make_chunk("c1", "Python programming language"),
            _make_chunk("c2", "Java programming language"),
            _make_chunk("c3", "Rust systems programming"),
            _make_chunk("c4", "Go concurrent programming"),
        ])

        result = store.search("Python", top_k=1)

        assert len(result) == 1
        assert result[0].score != 0.0
        assert result[0].chunk.chunk_id == "c1"

    def test_empty_query_returns_empty(self) -> None:
        store = BM25Store()
        store.add_chunks([_make_chunk("c1", "some content")])

        result = store.search("", top_k=5)

        assert result == ()

    def test_no_match_returns_empty(self) -> None:
        store = BM25Store()
        store.add_chunks([_make_chunk("c1", "apples and oranges")])

        result = store.search("quantum physics", top_k=5)

        assert result == ()

    def test_size_property(self) -> None:
        store = BM25Store()
        assert store.size == 0

        store.add_chunks([_make_chunk("c1", "content")])
        assert store.size == 1

    def test_clear(self) -> None:
        store = BM25Store()
        store.add_chunks([_make_chunk("c1", "content")])
        assert store.size == 1

        store.clear()
        assert store.size == 0
        assert store.search("content", top_k=5) == ()

    def test_incremental_add(self) -> None:
        store = BM25Store()
        store.add_chunks([_make_chunk("c1", "first batch Python")])
        store.add_chunks([_make_chunk("c2", "second batch Python")])

        result = store.search("Python", top_k=10)
        assert len(result) == 2


class TestBM25StoreTenantIsolation:
    def test_filters_by_tenant(self) -> None:
        store = BM25Store()
        store.add_chunks([
            _make_chunk("c1", "Python programming", tenant_id="tenant-a"),
            _make_chunk("c2", "Python scripting", tenant_id="tenant-b"),
        ])

        result_a = store.search("Python", top_k=10, tenant_id="tenant-a")
        assert len(result_a) == 1
        assert result_a[0].chunk.chunk_id == "c1"

        result_b = store.search("Python", top_k=10, tenant_id="tenant-b")
        assert len(result_b) == 1
        assert result_b[0].chunk.chunk_id == "c2"

    def test_no_tenant_returns_all(self) -> None:
        store = BM25Store()
        store.add_chunks([
            _make_chunk("c1", "Python programming", tenant_id="tenant-a"),
            _make_chunk("c2", "Python scripting", tenant_id="tenant-b"),
        ])

        result = store.search("Python", top_k=10)
        assert len(result) == 2

    def test_unknown_tenant_returns_empty(self) -> None:
        store = BM25Store()
        store.add_chunks([
            _make_chunk("c1", "Python programming", tenant_id="tenant-a"),
        ])

        result = store.search("Python", top_k=10, tenant_id="tenant-x")
        assert result == ()
