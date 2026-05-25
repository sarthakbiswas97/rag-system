from __future__ import annotations

import math

import pytest

from rag.ingestion.embedder import Embedder
from rag.models.document import Chunk, ChunkMetadata

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EXPECTED_DIM = 384


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder(model_name=MODEL_NAME, batch_size=32)


class TestEmbedTexts:
    def test_returns_correct_count(self, embedder: Embedder) -> None:
        result = embedder.embed_texts(["hello", "world", "foo"])
        assert len(result) == 3

    def test_returns_tuples_of_floats(self, embedder: Embedder) -> None:
        result = embedder.embed_texts(["test sentence"])
        assert isinstance(result[0], tuple)
        assert all(isinstance(x, float) for x in result[0])

    def test_correct_dimension(self, embedder: Embedder) -> None:
        result = embedder.embed_texts(["test sentence"])
        assert len(result[0]) == EXPECTED_DIM

    def test_identical_texts_produce_identical_embeddings(
        self, embedder: Embedder
    ) -> None:
        result = embedder.embed_texts(["same text", "same text"])
        assert result[0] == result[1]

    def test_different_texts_produce_different_embeddings(
        self, embedder: Embedder
    ) -> None:
        result = embedder.embed_texts(["cats are great", "quantum physics theory"])
        assert result[0] != result[1]

    def test_empty_input_returns_empty(self, embedder: Embedder) -> None:
        result = embedder.embed_texts([])
        assert result == ()

    def test_embeddings_are_normalized(self, embedder: Embedder) -> None:
        result = embedder.embed_texts(["test normalization"])
        l2_norm = math.sqrt(sum(x * x for x in result[0]))
        assert abs(l2_norm - 1.0) < 0.01


class TestEmbedChunks:
    def _make_chunk(self, text: str, doc_id: str = "doc-1") -> Chunk:
        return Chunk(
            document_id=doc_id,
            text=text,
            metadata=ChunkMetadata(
                source_file="test.txt",
                chunk_index=0,
                total_chunks=1,
            ),
        )

    def test_returns_chunks_with_embeddings(self, embedder: Embedder) -> None:
        chunks = [self._make_chunk("hello world")]
        result = embedder.embed_chunks(chunks)
        assert len(result) == 1
        assert result[0].embedding is not None
        assert len(result[0].embedding) == EXPECTED_DIM

    def test_original_chunks_unchanged(self, embedder: Embedder) -> None:
        original = self._make_chunk("hello world")
        embedder.embed_chunks([original])
        assert original.embedding is None

    def test_preserves_chunk_fields(self, embedder: Embedder) -> None:
        original = self._make_chunk("hello world", doc_id="my-doc")
        result = embedder.embed_chunks([original])
        embedded = result[0]
        assert embedded.document_id == "my-doc"
        assert embedded.text == "hello world"
        assert embedded.chunk_id == original.chunk_id
        assert embedded.metadata.source_file == "test.txt"

    def test_empty_input_returns_empty(self, embedder: Embedder) -> None:
        result = embedder.embed_chunks([])
        assert result == ()

    def test_multiple_chunks(self, embedder: Embedder) -> None:
        chunks = [self._make_chunk(f"text number {i}") for i in range(5)]
        result = embedder.embed_chunks(chunks)
        assert len(result) == 5
        ids = [c.chunk_id for c in result]
        assert len(ids) == len(set(ids))


class TestEmbedderProperties:
    def test_dimension_property(self, embedder: Embedder) -> None:
        assert embedder.dimension == EXPECTED_DIM
