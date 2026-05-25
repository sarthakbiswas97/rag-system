from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from rag.ingestion.embedder import Embedder
from rag.models.document import Chunk, ChunkMetadata
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
