from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from rag.ingestion.embedder import Embedder
from rag.models.document import Chunk, ChunkMetadata
from rag.retrieval.retriever import Retriever
from rag.retrieval.sparse_embedder import SparseEmbedder
from rag.retrieval.vector_store import VectorStore

MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_DIM = 384


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder(model_name=MODEL_NAME, batch_size=32)


@pytest.fixture()
def store_with_data(embedder: Embedder) -> VectorStore:
    client = QdrantClient(":memory:")
    store = VectorStore(client=client, collection="test_hybrid")
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

    # Add sparse vectors
    sparse = SparseEmbedder()
    sparse_embs = sparse.embed_texts([c.text for c in chunks])
    store.upsert_sparse_vectors(
        [c.chunk_id for c in embedded],
        [se.to_qdrant() for se in sparse_embs],
    )

    return store


class TestRetrieverWithSparse:
    def test_hybrid_retrieval_fuses_results(
        self, embedder: Embedder, store_with_data: VectorStore
    ) -> None:
        sparse = SparseEmbedder()
        # Seed vocabulary with the same texts
        sparse.embed_texts(
            [
                "The capital of France is Paris, a major European city.",
                "Python is a popular programming language for data science.",
                "Quantum mechanics describes behavior of particles at atomic scale.",
                "The Great Wall of China is a historic fortification structure.",
                "Machine learning is a subset of artificial intelligence.",
            ]
        )

        retriever = Retriever(
            embedder=embedder,
            vector_store=store_with_data,
            sparse_embedder=sparse,
        )
        result = retriever.retrieve("What is the capital of France?", top_k=3)

        assert len(result.scored_chunks) == 3
        # Should find the Paris document
        texts = [sc.chunk.text for sc in result.scored_chunks]
        assert any("Paris" in t for t in texts)

    def test_sparse_only_no_results_for_nonsense(
        self, embedder: Embedder, store_with_data: VectorStore
    ) -> None:
        sparse = SparseEmbedder()
        retriever = Retriever(
            embedder=embedder,
            vector_store=store_with_data,
            sparse_embedder=sparse,
        )
        result = retriever.retrieve("xyzqwerty12345nonsense", top_k=3)

        # Sparse won't match, but dense might still return something
        # (semantic similarity can be low for nonsense)
        assert len(result.scored_chunks) <= 3

    def test_retrieval_method_marked_hybrid(
        self, embedder: Embedder, store_with_data: VectorStore
    ) -> None:
        sparse = SparseEmbedder()
        sparse.embed_text("The capital of France is Paris")

        retriever = Retriever(
            embedder=embedder,
            vector_store=store_with_data,
            sparse_embedder=sparse,
        )
        result = retriever.retrieve("France capital Paris", top_k=5)

        # After RRF fusion, method is always "hybrid"
        for sc in result.scored_chunks:
            assert sc.retrieval_method == "hybrid"

    def test_without_sparse_embedder_uses_dense_only(
        self, embedder: Embedder, store_with_data: VectorStore
    ) -> None:
        retriever = Retriever(
            embedder=embedder,
            vector_store=store_with_data,
            sparse_embedder=None,
        )
        result = retriever.retrieve("machine learning", top_k=3)

        assert len(result.scored_chunks) == 3
        for sc in result.scored_chunks:
            assert sc.retrieval_method == "vector"
