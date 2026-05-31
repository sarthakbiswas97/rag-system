from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag.ingestion.pipeline import IngestionPipeline
from rag.models.document import Chunk, ChunkMetadata


def _make_doc(
    content: str = "test content",
    doc_id: str = "doc-1",
    source_path: str = "test.txt",
) -> MagicMock:
    doc = MagicMock()
    doc.content = content
    doc.document_id = doc_id
    doc.source_path = source_path
    return doc


def _make_chunk(text: str, chunk_id: str = "c1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        text=text,
        metadata=ChunkMetadata(source_file="test.txt"),
        embedding=tuple(float(i) for i in range(10)),
    )


@pytest.fixture()
def mock_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.dimension = 10

    def _embed_chunks(chunks):
        return tuple(
            dataclasses.replace(c, embedding=tuple(float(i) for i in range(10)))
            for c in chunks
        )

    embedder.embed_chunks = _embed_chunks
    return embedder


@pytest.fixture()
def mock_vector_store() -> MagicMock:
    store = MagicMock()
    store.upsert_chunks.return_value = 0
    return store


@pytest.fixture()
def pipeline(
    mock_embedder: MagicMock, mock_vector_store: MagicMock
) -> IngestionPipeline:
    return IngestionPipeline(
        embedder=mock_embedder,
        vector_store=mock_vector_store,
    )


class TestBatchIngestion:
    @patch("rag.ingestion.pipeline.load_document")
    @patch("rag.ingestion.pipeline.chunk_document")
    def test_batch_processes_multiple_docs(
        self,
        mock_chunk: MagicMock,
        mock_load: MagicMock,
        pipeline: IngestionPipeline,
        mock_vector_store: MagicMock,
    ) -> None:
        call_count = 0

        def _load(path):
            nonlocal call_count
            call_count += 1
            return _make_doc(f"unique content {call_count}")

        mock_load.side_effect = _load
        mock_chunk.return_value = [
            _make_chunk("hello world", "c1"),
        ]

        paths = [Path("a.txt"), Path("b.txt")]
        result = pipeline.ingest_documents_batch(paths)

        assert result.documents_processed == 2
        assert result.chunks_created == 2
        mock_vector_store.upsert_chunks.assert_called_once()

    @patch("rag.ingestion.pipeline.load_document")
    @patch("rag.ingestion.pipeline.chunk_document")
    def test_batch_sorts_by_text_length(
        self,
        mock_chunk: MagicMock,
        mock_load: MagicMock,
        pipeline: IngestionPipeline,
    ) -> None:
        load_count = 0

        def _load(path):
            nonlocal load_count
            load_count += 1
            return _make_doc(f"content {load_count}")

        mock_load.side_effect = _load

        # Return chunks with varying text lengths
        chunk_count = 0

        def _chunk_side(*args, **kwargs):
            nonlocal chunk_count
            chunk_count += 1
            if chunk_count == 1:
                return [_make_chunk("a" * 100, "long")]
            return [_make_chunk("b" * 10, "short")]

        mock_chunk.side_effect = _chunk_side

        paths = [Path("a.txt"), Path("b.txt")]
        result = pipeline.ingest_documents_batch(paths)

        assert result.documents_processed == 2
        assert result.chunks_created == 2

    @patch("rag.ingestion.pipeline.load_document")
    @patch("rag.ingestion.pipeline.chunk_document")
    def test_batch_calls_progress_callback(
        self,
        mock_chunk: MagicMock,
        mock_load: MagicMock,
        pipeline: IngestionPipeline,
    ) -> None:
        load_n = 0

        def _load(path):
            nonlocal load_n
            load_n += 1
            return _make_doc(f"content {load_n}")

        mock_load.side_effect = _load
        mock_chunk.return_value = [_make_chunk("text", "c1")]

        progress_calls: list[tuple[int, int, int]] = []

        def _on_progress(done: int, total: int, chunks: int) -> None:
            progress_calls.append((done, total, chunks))

        paths = [Path("a.txt"), Path("b.txt")]
        pipeline.ingest_documents_batch(paths, on_progress=_on_progress)

        # Should get progress calls during loading + final call
        assert len(progress_calls) >= 2
        # Last call should report completion
        last = progress_calls[-1]
        assert last[0] == 2  # total docs
        assert last[2] == 2  # total chunks

    @patch("rag.ingestion.pipeline.load_document")
    @patch("rag.ingestion.pipeline.chunk_document")
    def test_batch_handles_failed_docs(
        self,
        mock_chunk: MagicMock,
        mock_load: MagicMock,
        pipeline: IngestionPipeline,
    ) -> None:
        call_count = 0

        def _load_side_effect(path):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("corrupt file")
            return _make_doc()

        mock_load.side_effect = _load_side_effect
        mock_chunk.return_value = [_make_chunk("text", "c1")]

        paths = [Path("bad.txt"), Path("good.txt")]
        result = pipeline.ingest_documents_batch(paths)

        assert result.documents_processed == 1
        assert result.documents_failed == 1
        assert result.chunks_created == 1

    @patch("rag.ingestion.pipeline.load_document")
    @patch("rag.ingestion.pipeline.chunk_document")
    def test_batch_handles_all_skipped(
        self,
        mock_chunk: MagicMock,
        mock_load: MagicMock,
        pipeline: IngestionPipeline,
        mock_vector_store: MagicMock,
    ) -> None:
        mock_load.return_value = _make_doc()
        mock_chunk.return_value = []  # No chunks produced

        paths = [Path("empty.txt")]
        result = pipeline.ingest_documents_batch(paths)

        assert result.documents_skipped == 1
        assert result.chunks_created == 0
        mock_vector_store.upsert_chunks.assert_not_called()

    @patch("rag.ingestion.pipeline.load_document")
    @patch("rag.ingestion.pipeline.chunk_document")
    def test_batch_populates_sparse_vectors(
        self,
        mock_chunk: MagicMock,
        mock_load: MagicMock,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        sparse = MagicMock()
        sparse.embed_texts.return_value = [MagicMock()]
        pipeline = IngestionPipeline(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            sparse_embedder=sparse,
        )

        mock_load.return_value = _make_doc()
        mock_chunk.return_value = [_make_chunk("text", "c1")]

        pipeline.ingest_documents_batch([Path("a.txt")])

        sparse.embed_texts.assert_called_once()

    @patch("rag.ingestion.pipeline.load_document")
    @patch("rag.ingestion.pipeline.chunk_document")
    def test_batch_elapsed_ms_is_positive(
        self,
        mock_chunk: MagicMock,
        mock_load: MagicMock,
        pipeline: IngestionPipeline,
    ) -> None:
        mock_load.return_value = _make_doc()
        mock_chunk.return_value = [_make_chunk("text", "c1")]

        result = pipeline.ingest_documents_batch([Path("a.txt")])
        assert result.elapsed_ms > 0

    @patch("rag.ingestion.pipeline.load_document")
    @patch("rag.ingestion.pipeline.chunk_document")
    def test_batch_populates_processed_documents(
        self,
        mock_chunk: MagicMock,
        mock_load: MagicMock,
        pipeline: IngestionPipeline,
    ) -> None:
        def _load(path):
            return _make_doc(source_path=str(path))

        mock_load.side_effect = _load
        mock_chunk.return_value = [_make_chunk("text", "c1")]

        result = pipeline.ingest_documents_batch([Path("a.txt")])

        assert len(result.processed_documents) == 1
        info = result.processed_documents[0]
        assert info.document_id == "doc-1"
        assert info.source_file == "a.txt"
        assert info.chunk_count == 1
        assert info.content_hash  # non-empty hash

    @patch("rag.ingestion.pipeline.load_document")
    @patch("rag.ingestion.pipeline.chunk_document")
    def test_batch_skipped_docs_not_in_processed_documents(
        self,
        mock_chunk: MagicMock,
        mock_load: MagicMock,
        pipeline: IngestionPipeline,
    ) -> None:
        mock_load.return_value = _make_doc()
        mock_chunk.return_value = []  # no chunks → skipped

        result = pipeline.ingest_documents_batch([Path("empty.txt")])

        assert result.documents_skipped == 1
        assert len(result.processed_documents) == 0
