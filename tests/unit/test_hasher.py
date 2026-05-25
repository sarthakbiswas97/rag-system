from __future__ import annotations

import dataclasses

import pytest

from rag.ingestion.hasher import ContentHasher, compute_hash
from rag.models.document import RawDocument
from rag.models.ingestion import IngestionResult


class TestComputeHash:
    def test_same_text_same_hash(self) -> None:
        assert compute_hash("hello world") == compute_hash("hello world")

    def test_different_text_different_hash(self) -> None:
        assert compute_hash("hello") != compute_hash("world")

    def test_normalizes_whitespace(self) -> None:
        assert compute_hash("hello   world") == compute_hash("hello world")

    def test_normalizes_case(self) -> None:
        assert compute_hash("Hello World") == compute_hash("hello world")

    def test_normalizes_newlines(self) -> None:
        assert compute_hash("hello\n\nworld") == compute_hash("hello world")

    def test_strips_leading_trailing(self) -> None:
        assert compute_hash("  hello  ") == compute_hash("hello")

    def test_empty_string(self) -> None:
        h = compute_hash("")
        assert isinstance(h, str)
        assert len(h) == 64  # sha256 hex length


class TestContentHasher:
    def _make_doc(self, content: str) -> RawDocument:
        return RawDocument(content=content, source_path="test.txt", file_type=".txt")

    def test_first_doc_not_duplicate(self) -> None:
        hasher = ContentHasher()
        assert not hasher.is_duplicate(self._make_doc("hello"))

    def test_same_content_is_duplicate(self) -> None:
        hasher = ContentHasher()
        doc = self._make_doc("hello")
        hasher.is_duplicate(doc)
        assert hasher.is_duplicate(self._make_doc("hello"))

    def test_different_content_not_duplicate(self) -> None:
        hasher = ContentHasher()
        hasher.is_duplicate(self._make_doc("hello"))
        assert not hasher.is_duplicate(self._make_doc("world"))

    def test_reset_clears_state(self) -> None:
        hasher = ContentHasher()
        hasher.is_duplicate(self._make_doc("hello"))
        hasher.reset()
        assert not hasher.is_duplicate(self._make_doc("hello"))

    def test_seen_count(self) -> None:
        hasher = ContentHasher()
        assert hasher.seen_count == 0
        hasher.is_duplicate(self._make_doc("a"))
        hasher.is_duplicate(self._make_doc("b"))
        assert hasher.seen_count == 2
        # duplicate doesn't increase count
        hasher.is_duplicate(self._make_doc("a"))
        assert hasher.seen_count == 2


class TestIngestionResult:
    def test_creates(self) -> None:
        result = IngestionResult(
            documents_processed=5,
            documents_skipped=2,
            documents_failed=1,
            chunks_created=30,
            elapsed_ms=1234.5,
        )
        assert result.documents_processed == 5
        assert result.chunks_created == 30

    def test_is_frozen(self) -> None:
        result = IngestionResult(
            documents_processed=0,
            documents_skipped=0,
            documents_failed=0,
            chunks_created=0,
            elapsed_ms=0.0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.documents_processed = 1  # type: ignore[misc]
