from __future__ import annotations

from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from rag.ingestion.embedder import Embedder
from rag.ingestion.pipeline import IngestionPipeline
from rag.retrieval.vector_store import VectorStore

MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_DIM = 384


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder(model_name=MODEL_NAME, batch_size=32)


@pytest.fixture()
def vector_store() -> VectorStore:
    client = QdrantClient(":memory:")
    store = VectorStore(client=client, collection="test_pipeline")
    store.create_collection(vector_size=VECTOR_DIM)
    return store


@pytest.fixture()
def pipeline(embedder: Embedder, vector_store: VectorStore) -> IngestionPipeline:
    return IngestionPipeline(
        embedder=embedder,
        vector_store=vector_store,
        chunk_size=50,
        chunk_overlap=5,
    )


def _create_txt_files(tmp_path: Path, contents: dict[str, str]) -> list[Path]:
    paths = []
    for name, text in contents.items():
        p = tmp_path / name
        p.write_text(text, encoding="utf-8")
        paths.append(p)
    return paths


class TestIngestDocuments:
    def test_ingests_multiple_files(
        self, pipeline: IngestionPipeline, vector_store: VectorStore, tmp_path: Path
    ) -> None:
        paths = _create_txt_files(
            tmp_path,
            {
                "a.txt": "Alpha bravo charlie delta echo foxtrot golf.",
                "b.txt": "Hotel india juliet kilo lima mike november.",
                "c.txt": "Oscar papa quebec romeo sierra tango uniform.",
            },
        )
        result = pipeline.ingest_documents(paths)
        assert result.documents_processed == 3
        assert result.documents_failed == 0
        assert result.chunks_created > 0
        assert vector_store.count() == result.chunks_created

    def test_deduplication_skips_same_content(
        self, pipeline: IngestionPipeline, tmp_path: Path
    ) -> None:
        paths = _create_txt_files(
            tmp_path,
            {"a.txt": "Same content here for testing dedup."},
        )
        result1 = pipeline.ingest_documents(paths)
        assert result1.documents_processed == 1

        result2 = pipeline.ingest_documents(paths)
        assert result2.documents_processed == 0
        assert result2.documents_skipped == 1

    def test_bad_file_does_not_crash_batch(
        self, pipeline: IngestionPipeline, tmp_path: Path
    ) -> None:
        good = tmp_path / "good.txt"
        good.write_text("This is valid content for ingestion.", encoding="utf-8")
        bad = tmp_path / "bad.csv"
        bad.write_text("not,supported", encoding="utf-8")

        result = pipeline.ingest_documents([good, bad])
        assert result.documents_processed == 1
        assert result.documents_failed == 1

    def test_empty_paths_returns_zeros(self, pipeline: IngestionPipeline) -> None:
        result = pipeline.ingest_documents([])
        assert result.documents_processed == 0
        assert result.documents_skipped == 0
        assert result.documents_failed == 0
        assert result.chunks_created == 0

    def test_empty_content_skipped(
        self, pipeline: IngestionPipeline, tmp_path: Path
    ) -> None:
        paths = _create_txt_files(tmp_path, {"empty.txt": ""})
        result = pipeline.ingest_documents(paths)
        assert result.documents_processed == 0
        assert result.documents_skipped == 1


class TestIngestDirectory:
    def test_finds_supported_files(
        self, pipeline: IngestionPipeline, vector_store: VectorStore, tmp_path: Path
    ) -> None:
        _create_txt_files(
            tmp_path,
            {
                "doc1.txt": "First document with enough words to chunk.",
                "doc2.md": "Second document in markdown format here.",
                "ignore.csv": "should,be,ignored",
            },
        )
        result = pipeline.ingest_directory(tmp_path)
        assert result.documents_processed == 2
        assert result.documents_failed == 0
        assert vector_store.count() > 0

    def test_empty_directory(self, pipeline: IngestionPipeline, tmp_path: Path) -> None:
        result = pipeline.ingest_directory(tmp_path)
        assert result.documents_processed == 0
        assert result.chunks_created == 0

    def test_raises_on_non_directory(
        self, pipeline: IngestionPipeline, tmp_path: Path
    ) -> None:
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("hello", encoding="utf-8")
        with pytest.raises(NotADirectoryError):
            pipeline.ingest_directory(file_path)
