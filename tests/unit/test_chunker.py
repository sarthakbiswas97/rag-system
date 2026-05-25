from __future__ import annotations

import dataclasses

import pytest

from rag.ingestion.chunker import (
    DEFAULT_SEPARATORS,
    _split_text,
    _token_count,
    chunk_document,
)
from rag.models.document import RawDocument


class TestTokenCount:
    def test_counts_words(self) -> None:
        assert _token_count("hello world foo") == 3

    def test_empty_string(self) -> None:
        assert _token_count("") == 0

    def test_single_word(self) -> None:
        assert _token_count("hello") == 1


class TestSplitText:
    def test_short_text_returns_single_chunk(self) -> None:
        result = _split_text("hello world", 10, 2, DEFAULT_SEPARATORS)
        assert result == ["hello world"]

    def test_empty_text_returns_empty(self) -> None:
        result = _split_text("", 10, 2, DEFAULT_SEPARATORS)
        assert result == []

    def test_whitespace_only_returns_empty(self) -> None:
        result = _split_text("   \n\n  ", 10, 2, DEFAULT_SEPARATORS)
        assert result == []

    def test_splits_on_paragraph_boundary(self) -> None:
        para1 = "First paragraph with several words."
        para2 = "Second paragraph with more words."
        text = f"{para1}\n\n{para2}"
        result = _split_text(text, 6, 1, DEFAULT_SEPARATORS)
        assert len(result) == 2
        assert "First paragraph" in result[0]
        assert "Second paragraph" in result[1]

    def test_splits_on_newline_when_paragraphs_too_large(self) -> None:
        text = "line one\nline two\nline three\nline four"
        result = _split_text(text, 3, 0, DEFAULT_SEPARATORS)
        assert len(result) >= 2
        for chunk in result:
            assert _token_count(chunk) <= 4  # small tolerance

    def test_splits_on_sentence_boundary(self) -> None:
        text = "First sentence here. Second sentence here. Third sentence here."
        result = _split_text(text, 4, 0, DEFAULT_SEPARATORS)
        assert len(result) >= 2

    def test_hard_split_on_words_as_last_resort(self) -> None:
        # no paragraph or sentence boundaries, just words
        words = " ".join(f"word{i}" for i in range(20))
        result = _split_text(words, 5, 0, DEFAULT_SEPARATORS)
        assert len(result) >= 3
        for chunk in result:
            assert _token_count(chunk) <= 5

    def test_overlap_between_chunks(self) -> None:
        text = "A B C D E F G H I J K L M N O P"
        result = _split_text(text, 5, 2, [" ", ""])
        assert len(result) >= 2
        # check that consecutive chunks share some content
        for i in range(len(result) - 1):
            words_current = set(result[i].split())
            words_next = set(result[i + 1].split())
            overlap = words_current & words_next
            assert len(overlap) > 0, f"No overlap between chunk {i} and {i + 1}"

    def test_no_empty_chunks_in_output(self) -> None:
        text = "Hello\n\n\n\nWorld\n\n\n\nFoo"
        result = _split_text(text, 5, 0, DEFAULT_SEPARATORS)
        for chunk in result:
            assert chunk.strip() != ""


class TestChunkDocument:
    def _make_doc(self, content: str, source: str = "test.txt") -> RawDocument:
        return RawDocument(
            content=content,
            source_path=source,
            file_type=".txt",
        )

    def test_empty_document_returns_empty_tuple(self) -> None:
        doc = self._make_doc("")
        assert chunk_document(doc) == ()

    def test_whitespace_document_returns_empty_tuple(self) -> None:
        doc = self._make_doc("   \n\n  ")
        assert chunk_document(doc) == ()

    def test_short_document_returns_single_chunk(self) -> None:
        doc = self._make_doc("Hello world.")
        chunks = chunk_document(doc, chunk_size=512, chunk_overlap=64)
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world."

    def test_propagates_document_id(self) -> None:
        doc = self._make_doc("Some content here.")
        chunks = chunk_document(doc)
        for chunk in chunks:
            assert chunk.document_id == doc.document_id

    def test_propagates_source_file(self) -> None:
        doc = self._make_doc("Some content here.", source="data/report.pdf")
        chunks = chunk_document(doc)
        for chunk in chunks:
            assert chunk.metadata.source_file == "data/report.pdf"

    def test_chunk_indices_are_sequential(self) -> None:
        words = " ".join(f"word{i}" for i in range(100))
        doc = self._make_doc(words)
        chunks = chunk_document(doc, chunk_size=10, chunk_overlap=2)
        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            assert chunk.metadata.chunk_index == i
            assert chunk.metadata.total_chunks == len(chunks)

    def test_all_chunks_are_frozen(self) -> None:
        doc = self._make_doc("Hello world and more text.")
        chunks = chunk_document(doc)
        for chunk in chunks:
            with pytest.raises(dataclasses.FrozenInstanceError):
                chunk.text = "modified"  # type: ignore[misc]

    def test_unique_chunk_ids(self) -> None:
        words = " ".join(f"word{i}" for i in range(100))
        doc = self._make_doc(words)
        chunks = chunk_document(doc, chunk_size=10, chunk_overlap=2)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_raises_on_overlap_gte_chunk_size(self) -> None:
        doc = self._make_doc("Hello world.")
        with pytest.raises(ValueError, match="chunk_overlap.*must be less than"):
            chunk_document(doc, chunk_size=10, chunk_overlap=10)

    def test_custom_separators(self) -> None:
        text = "alpha bravo|charlie delta|echo foxtrot"
        doc = self._make_doc(text)
        chunks = chunk_document(
            doc, chunk_size=2, chunk_overlap=0, separators=["|", " ", ""]
        )
        assert len(chunks) == 3

    def test_parent_chunk_id_defaults_to_none(self) -> None:
        doc = self._make_doc("Some content.")
        chunks = chunk_document(doc)
        for chunk in chunks:
            assert chunk.metadata.parent_chunk_id is None

    def test_multi_paragraph_document(self) -> None:
        paragraphs = [" ".join(f"p{p}word{w}" for w in range(20)) for p in range(5)]
        text = "\n\n".join(paragraphs)
        doc = self._make_doc(text)
        chunks = chunk_document(doc, chunk_size=25, chunk_overlap=3)
        assert len(chunks) >= 3
        total_words_in_chunks = sum(_token_count(c.text) for c in chunks)
        total_words_in_doc = _token_count(text)
        # with overlap, total chunk words >= doc words
        assert total_words_in_chunks >= total_words_in_doc
