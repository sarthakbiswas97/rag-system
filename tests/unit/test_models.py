import dataclasses

import pytest

from rag.models.document import Chunk, ChunkMetadata, RawDocument
from rag.models.generation import Citation, GenerationResponse
from rag.models.retrieval import RetrievalResult, ScoredChunk
from rag.models.session import ConversationTurn
from rag.models.verification import EntailmentResult, VerificationReport


class TestChunkMetadata:
    def test_creates_with_defaults(self) -> None:
        meta = ChunkMetadata(source_file="test.txt")
        assert meta.source_file == "test.txt"
        assert meta.page_number is None
        assert meta.chunk_index == 0
        assert meta.created_at  # non-empty timestamp

    def test_parent_chunk_id_defaults_to_none(self) -> None:
        meta = ChunkMetadata(source_file="test.txt")
        assert meta.parent_chunk_id is None

    def test_parent_chunk_id_stores_value(self) -> None:
        meta = ChunkMetadata(source_file="test.txt", parent_chunk_id="parent-123")
        assert meta.parent_chunk_id == "parent-123"

    def test_is_frozen(self) -> None:
        meta = ChunkMetadata(source_file="test.txt")
        with pytest.raises(dataclasses.FrozenInstanceError):
            meta.source_file = "other.txt"  # type: ignore[misc]


class TestChunk:
    def test_creates_with_unique_ids(self) -> None:
        chunk_a = Chunk(text="hello")
        chunk_b = Chunk(text="world")
        assert chunk_a.chunk_id != chunk_b.chunk_id

    def test_is_frozen(self) -> None:
        chunk = Chunk(text="hello")
        with pytest.raises(dataclasses.FrozenInstanceError):
            chunk.text = "modified"  # type: ignore[misc]

    def test_embedding_default_is_none(self) -> None:
        chunk = Chunk(text="hello")
        assert chunk.embedding is None

    def test_stores_embedding_as_tuple(self) -> None:
        chunk = Chunk(text="hello", embedding=(0.1, 0.2, 0.3))
        assert chunk.embedding == (0.1, 0.2, 0.3)


class TestRawDocument:
    def test_creates_with_unique_ids(self) -> None:
        doc_a = RawDocument(content="a")
        doc_b = RawDocument(content="b")
        assert doc_a.document_id != doc_b.document_id

    def test_is_frozen(self) -> None:
        doc = RawDocument(content="text")
        with pytest.raises(dataclasses.FrozenInstanceError):
            doc.content = "modified"  # type: ignore[misc]


class TestScoredChunk:
    def test_creates_with_score(self) -> None:
        chunk = Chunk(text="hello")
        scored = ScoredChunk(chunk=chunk, score=0.95, retrieval_method="vector")
        assert scored.score == 0.95
        assert scored.retrieval_method == "vector"

    def test_is_frozen(self) -> None:
        scored = ScoredChunk(
            chunk=Chunk(text="hello"), score=0.95, retrieval_method="vector"
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            scored.score = 0.5  # type: ignore[misc]


class TestRetrievalResult:
    def test_creates_with_empty_chunks(self) -> None:
        result = RetrievalResult(
            query="test",
            rewritten_query=None,
            scored_chunks=(),
            retrieval_time_ms=10.0,
        )
        assert result.scored_chunks == ()
        assert result.rewritten_query is None


class TestCitation:
    def test_creates(self) -> None:
        citation = Citation(
            chunk_id="abc",
            source_file="doc.pdf",
            text_snippet="some text",
            sentence_index=0,
        )
        assert citation.source_file == "doc.pdf"


class TestGenerationResponse:
    def test_abstention(self) -> None:
        result = RetrievalResult(
            query="test",
            rewritten_query=None,
            scored_chunks=(),
            retrieval_time_ms=5.0,
        )
        response = GenerationResponse(
            answer="I don't have enough information.",
            citations=(),
            is_abstention=True,
            confidence_score=0.0,
            retrieval_result=result,
            generation_time_ms=100.0,
        )
        assert response.is_abstention is True
        assert response.citations == ()


class TestEntailmentResult:
    def test_creates(self) -> None:
        result = EntailmentResult(
            sentence="Paris is the capital.",
            label="entailment",
            confidence=0.98,
            supporting_chunk_id="chunk-1",
        )
        assert result.label == "entailment"


class TestVerificationReport:
    def test_faithful_report(self) -> None:
        report = VerificationReport(
            sentence_results=(),
            overall_faithful=True,
            faithfulness_score=1.0,
        )
        assert report.overall_faithful is True


class TestConversationTurn:
    def test_creates_with_timestamp(self) -> None:
        turn = ConversationTurn(role="user", content="hello")
        assert turn.role == "user"
        assert turn.timestamp  # non-empty

    def test_is_frozen(self) -> None:
        turn = ConversationTurn(role="user", content="hello")
        with pytest.raises(dataclasses.FrozenInstanceError):
            turn.content = "modified"  # type: ignore[misc]
