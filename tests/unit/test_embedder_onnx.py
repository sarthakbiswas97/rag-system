from __future__ import annotations

import pytest

from rag.ingestion.embedder import Embedder

MODEL_NAME = "BAAI/bge-small-en-v1.5"


class TestEmbedderOnnx:
    def test_pytorch_backend_default(self) -> None:
        embedder = Embedder(model_name=MODEL_NAME, backend="pytorch")
        assert embedder.dimension == 384
        texts = ["Hello world", "Test sentence"]
        embeddings = embedder.embed_texts(texts)
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 384

    @pytest.mark.slow
    def test_onnx_backend(self) -> None:
        # ONNX export can be slow on first run
        embedder = Embedder(
            model_name=MODEL_NAME,
            backend="onnx",
            onnx_provider="CPUExecutionProvider",
        )
        assert embedder.dimension == 384
        texts = ["Hello world", "Test sentence"]
        embeddings = embedder.embed_texts(texts)
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 384

    def test_embed_chunks(self) -> None:
        from rag.models.document import Chunk, ChunkMetadata

        embedder = Embedder(model_name=MODEL_NAME, backend="pytorch")
        chunks = [
            Chunk(
                chunk_id="c1",
                document_id="doc-1",
                text="Hello world",
                metadata=ChunkMetadata(source_file="test.txt"),
            ),
            Chunk(
                chunk_id="c2",
                document_id="doc-1",
                text="Another sentence",
                metadata=ChunkMetadata(source_file="test.txt"),
            ),
        ]
        result = embedder.embed_chunks(chunks)
        assert len(result) == 2
        assert result[0].embedding is not None
        assert len(result[0].embedding) == 384

    def test_empty_texts(self) -> None:
        embedder = Embedder(model_name=MODEL_NAME, backend="pytorch")
        assert embedder.embed_texts([]) == ()

    def test_empty_chunks(self) -> None:
        embedder = Embedder(model_name=MODEL_NAME, backend="pytorch")
        assert embedder.embed_chunks([]) == ()
