from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from qdrant_client import QdrantClient

from rag.ingestion.embedder import Embedder
from rag.models.document import Chunk, ChunkMetadata
from rag.retrieval.reranker import Reranker
from rag.retrieval.retriever import Retriever
from rag.retrieval.vector_store import VectorStore

MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_DIM = 384


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder(model_name=MODEL_NAME, batch_size=32)


@pytest.fixture()
def store_with_data(embedder: Embedder) -> VectorStore:
    client = QdrantClient(":memory:")
    store = VectorStore(client=client, collection="test_retriever")
    store.create_collection(vector_size=VECTOR_DIM)

    chunks = [
        Chunk(
            document_id="doc-1",
            text=text,
            metadata=ChunkMetadata(source_file="test.txt"),
        )
        for text in [
            "The capital of France is Paris, a major European city.",
            "Python is a popular programming language for data science.",
            "Quantum mechanics describes behavior of particles at atomic scale.",
            "The Great Wall of China is a historic fortification structure.",
            "Machine learning is a subset of artificial intelligence.",
        ]
    ]
    embedded = embedder.embed_chunks(chunks)
    store.upsert_chunks(embedded)
    return store


@pytest.fixture()
def empty_store() -> VectorStore:
    client = QdrantClient(":memory:")
    store = VectorStore(client=client, collection="test_empty")
    store.create_collection(vector_size=VECTOR_DIM)
    return store


class TestRetriever:
    def test_retrieves_relevant_result(
        self, embedder: Embedder, store_with_data: VectorStore
    ) -> None:
        retriever = Retriever(embedder=embedder, vector_store=store_with_data)
        result = retriever.retrieve("What is the capital of France?", top_k=3)

        assert result.query == "What is the capital of France?"
        assert result.rewritten_query is None
        assert len(result.scored_chunks) == 3
        assert "Paris" in result.scored_chunks[0].chunk.text

    def test_empty_collection_returns_empty(
        self, embedder: Embedder, empty_store: VectorStore
    ) -> None:
        retriever = Retriever(embedder=embedder, vector_store=empty_store)
        result = retriever.retrieve("anything", top_k=5)

        assert result.scored_chunks == ()
        assert result.retrieval_time_ms >= 0

    def test_top_k_limits_results(
        self, embedder: Embedder, store_with_data: VectorStore
    ) -> None:
        retriever = Retriever(embedder=embedder, vector_store=store_with_data)
        result = retriever.retrieve("programming", top_k=2)

        assert len(result.scored_chunks) == 2

    def test_retrieval_method_is_vector(
        self, embedder: Embedder, store_with_data: VectorStore
    ) -> None:
        retriever = Retriever(embedder=embedder, vector_store=store_with_data)
        result = retriever.retrieve("machine learning", top_k=3)

        for sc in result.scored_chunks:
            assert sc.retrieval_method == "vector"

    def test_timing_is_recorded(
        self, embedder: Embedder, store_with_data: VectorStore
    ) -> None:
        retriever = Retriever(embedder=embedder, vector_store=store_with_data)
        result = retriever.retrieve("France", top_k=1)

        assert result.retrieval_time_ms >= 0


class TestRetrieverWithReranker:
    @pytest.fixture()
    def mock_reranker(self) -> MagicMock:
        reranker = MagicMock(spec=Reranker)
        return reranker

    def test_reranker_called_when_provided(
        self,
        embedder: Embedder,
        store_with_data: VectorStore,
        mock_reranker: MagicMock,
    ) -> None:
        mock_reranker.rerank.return_value = ()
        retriever = Retriever(
            embedder=embedder,
            vector_store=store_with_data,
            reranker=mock_reranker,
        )
        retriever.retrieve("France", top_k=3)

        mock_reranker.rerank.assert_called_once()
        call_kwargs = mock_reranker.rerank.call_args
        assert call_kwargs[1]["top_k"] == 3

    def test_reranker_not_called_when_none(
        self,
        embedder: Embedder,
        store_with_data: VectorStore,
    ) -> None:
        retriever = Retriever(
            embedder=embedder,
            vector_store=store_with_data,
            reranker=None,
        )
        result = retriever.retrieve("France", top_k=3)

        assert len(result.scored_chunks) == 3
        for sc in result.scored_chunks:
            assert sc.retrieval_method == "vector"

    def test_broader_search_when_reranking(
        self,
        embedder: Embedder,
        store_with_data: VectorStore,
        mock_reranker: MagicMock,
    ) -> None:
        mock_reranker.rerank.return_value = ()
        retriever = Retriever(
            embedder=embedder,
            vector_store=store_with_data,
            reranker=mock_reranker,
        )
        # With 5 docs in store and top_k=2, reranker path should search
        # max(2*10, 50) = 50, but only 5 docs exist so we get 5
        retriever.retrieve("France", top_k=2)

        chunks_passed = mock_reranker.rerank.call_args[0][1]
        assert len(chunks_passed) == 5  # all docs retrieved before reranking


class TestRetrieverTenantIsolation:
    def test_retrieve_filters_by_tenant(self, embedder: Embedder) -> None:
        client = QdrantClient(":memory:")
        store = VectorStore(client=client, collection="test_tenant_isolation")
        store.create_collection(vector_size=VECTOR_DIM)

        chunks_a = [
            Chunk(
                document_id="doc-a",
                text="Paris is the capital of France.",
                metadata=ChunkMetadata(source_file="a.txt", tenant_id="tenant-a"),
            )
        ]
        chunks_b = [
            Chunk(
                document_id="doc-b",
                text="Berlin is the capital of Germany.",
                metadata=ChunkMetadata(source_file="b.txt", tenant_id="tenant-b"),
            )
        ]
        store.upsert_chunks(embedder.embed_chunks(chunks_a))
        store.upsert_chunks(embedder.embed_chunks(chunks_b))

        retriever = Retriever(embedder=embedder, vector_store=store)

        result_a = retriever.retrieve("capital", top_k=10, tenant_id="tenant-a")
        assert len(result_a.scored_chunks) == 1
        assert result_a.scored_chunks[0].chunk.metadata.tenant_id == "tenant-a"

        result_b = retriever.retrieve("capital", top_k=10, tenant_id="tenant-b")
        assert len(result_b.scored_chunks) == 1
        assert result_b.scored_chunks[0].chunk.metadata.tenant_id == "tenant-b"

    def test_retrieve_without_tenant_returns_all(self, embedder: Embedder) -> None:
        client = QdrantClient(":memory:")
        store = VectorStore(client=client, collection="test_no_filter")
        store.create_collection(vector_size=VECTOR_DIM)

        chunks = [
            Chunk(
                document_id="doc-a",
                text="Paris is the capital of France.",
                metadata=ChunkMetadata(source_file="a.txt", tenant_id="tenant-a"),
            ),
            Chunk(
                document_id="doc-b",
                text="Berlin is the capital of Germany.",
                metadata=ChunkMetadata(source_file="b.txt", tenant_id="tenant-b"),
            ),
        ]
        store.upsert_chunks(embedder.embed_chunks(chunks))

        retriever = Retriever(embedder=embedder, vector_store=store)
        result = retriever.retrieve("capital", top_k=10)
        assert len(result.scored_chunks) == 2
