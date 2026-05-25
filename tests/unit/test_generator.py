from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from rag.generation.generator import (
    Generator,
    compute_confidence,
    detect_abstention,
    parse_citations,
)
from rag.models.document import Chunk, ChunkMetadata
from rag.models.generation import LLMResponse
from rag.models.retrieval import RetrievalResult, ScoredChunk


def _make_sc(
    text: str = "chunk text",
    source: str = "doc.pdf",
    score: float = 0.9,
    chunk_id: str | None = None,
) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=chunk_id or f"id-{text[:8]}",
        document_id="doc-1",
        text=text,
        metadata=ChunkMetadata(source_file=source),
    )
    return ScoredChunk(chunk=chunk, score=score, retrieval_method="vector")


# --- parse_citations ---


class TestParseCitations:
    def test_single_citation(self) -> None:
        chunks = [_make_sc("Paris info", "geo.pdf")]
        result = parse_citations("Paris is the capital [1].", chunks)
        assert len(result) == 1
        assert result[0].chunk_id == "id-Paris in"
        assert result[0].source_file == "geo.pdf"
        assert result[0].sentence_index == 0

    def test_two_sentences_two_citations(self) -> None:
        chunks = [_make_sc("Alpha"), _make_sc("Bravo")]
        result = parse_citations("Alpha fact [1]. Bravo fact [2].", chunks)
        assert len(result) == 2
        assert result[0].sentence_index == 0
        assert result[1].sentence_index == 1

    def test_multiple_citations_same_sentence(self) -> None:
        chunks = [_make_sc("A"), _make_sc("B"), _make_sc("C")]
        result = parse_citations("Combined fact [1][3].", chunks)
        assert len(result) == 2

    def test_no_citations(self) -> None:
        chunks = [_make_sc("text")]
        result = parse_citations("No citations here.", chunks)
        assert result == ()

    def test_invalid_index_skipped(self) -> None:
        chunks = [_make_sc("A"), _make_sc("B")]
        result = parse_citations("Invalid [99].", chunks)
        assert result == ()

    def test_index_zero_skipped(self) -> None:
        chunks = [_make_sc("A")]
        result = parse_citations("[0] text.", chunks)
        assert result == ()

    def test_duplicate_same_sentence_deduped(self) -> None:
        chunks = [_make_sc("A", chunk_id="chunk-1")]
        result = parse_citations("Text [1][1].", chunks)
        assert len(result) == 1

    def test_citation_in_middle_of_word(self) -> None:
        chunks = [_make_sc("A")]
        result = parse_citations("word[1]more", chunks)
        assert len(result) == 1

    def test_snippet_truncated(self) -> None:
        long_text = "x" * 500
        chunks = [_make_sc(long_text)]
        result = parse_citations("Fact [1].", chunks)
        assert len(result[0].text_snippet) == 200

    def test_empty_text_returns_empty(self) -> None:
        chunks = [_make_sc("A")]
        assert parse_citations("", chunks) == ()

    def test_empty_chunks_returns_empty(self) -> None:
        assert parse_citations("Text [1].", []) == ()


# --- detect_abstention ---


class TestDetectAbstention:
    def test_abstention_phrase(self) -> None:
        assert detect_abstention(
            "I don't have enough information to answer.", has_citations=False
        )

    def test_not_enough_info(self) -> None:
        assert detect_abstention(
            "There is not enough information.", has_citations=False
        )

    def test_normal_answer(self) -> None:
        assert not detect_abstention(
            "Paris is the capital of France.", has_citations=False
        )

    def test_case_insensitive(self) -> None:
        assert detect_abstention(
            "I DON'T HAVE ENOUGH INFORMATION.", has_citations=False
        )

    def test_partial_answer_with_citations_not_abstention(self) -> None:
        assert not detect_abstention(
            "I don't have enough information to fully answer, but [1]...",
            has_citations=True,
        )

    def test_empty_response_is_abstention(self) -> None:
        assert detect_abstention("", has_citations=False)

    def test_whitespace_only_is_abstention(self) -> None:
        assert detect_abstention("   \n  ", has_citations=False)

    def test_cannot_answer(self) -> None:
        assert detect_abstention("I cannot answer this question.", has_citations=False)


# --- compute_confidence ---


class TestComputeConfidence:
    def test_average_scores(self) -> None:
        chunks = [_make_sc(score=0.9), _make_sc(score=0.8), _make_sc(score=0.7)]
        assert abs(compute_confidence(chunks) - 0.8) < 0.001

    def test_empty_returns_zero(self) -> None:
        assert compute_confidence([]) == 0.0

    def test_clamps_above_one(self) -> None:
        chunks = [_make_sc(score=1.5)]
        assert compute_confidence(chunks) == 1.0

    def test_clamps_negative(self) -> None:
        chunks = [_make_sc(score=-0.1)]
        assert compute_confidence(chunks) == 0.0

    def test_single_chunk(self) -> None:
        chunks = [_make_sc(score=0.75)]
        assert compute_confidence(chunks) == 0.75


# --- Generator ---


def _make_retrieval_result(
    chunks: list[ScoredChunk] | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        query="test question",
        rewritten_query=None,
        scored_chunks=tuple(chunks or []),
        retrieval_time_ms=10.0,
    )


def _make_llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        prompt_tokens=50,
        completion_tokens=20,
        model="gpt-4o-mini",
        elapsed_ms=500.0,
    )


class TestGenerator:
    @pytest.fixture()
    def mock_llm(self) -> MagicMock:
        client = MagicMock()
        client.generate = AsyncMock()
        return client

    @pytest.fixture()
    def generator(self, mock_llm: MagicMock) -> Generator:
        return Generator(llm_client=mock_llm)

    @pytest.mark.asyncio
    async def test_generates_with_citations(
        self, generator: Generator, mock_llm: MagicMock
    ) -> None:
        chunks = [_make_sc("Paris info", "geo.pdf", 0.95)]
        retrieval = _make_retrieval_result(chunks)
        mock_llm.generate.return_value = _make_llm_response("Paris is the capital [1].")

        result = await generator.generate("What is the capital?", retrieval)

        assert result.answer == "Paris is the capital [1]."
        assert len(result.citations) == 1
        assert result.is_abstention is False
        assert result.confidence_score == 0.95
        assert result.generation_time_ms == 500.0

    @pytest.mark.asyncio
    async def test_abstention_response(
        self, generator: Generator, mock_llm: MagicMock
    ) -> None:
        chunks = [_make_sc("Unrelated", score=0.2)]
        retrieval = _make_retrieval_result(chunks)
        mock_llm.generate.return_value = _make_llm_response(
            "I don't have enough information in the available sources "
            "to answer this question."
        )

        result = await generator.generate("Unknown topic?", retrieval)

        assert result.is_abstention is True
        assert result.citations == ()

    @pytest.mark.asyncio
    async def test_top_k_slicing(
        self, generator: Generator, mock_llm: MagicMock
    ) -> None:
        chunks = [_make_sc(f"chunk-{i}", score=0.9 - i * 0.1) for i in range(10)]
        retrieval = _make_retrieval_result(chunks)
        mock_llm.generate.return_value = _make_llm_response("Answer [1].")

        await generator.generate("question", retrieval, top_k=3)

        call_args = mock_llm.generate.call_args
        user_prompt = call_args.args[1]
        assert "[4]" not in user_prompt

    @pytest.mark.asyncio
    async def test_top_k_larger_than_available(
        self, generator: Generator, mock_llm: MagicMock
    ) -> None:
        chunks = [_make_sc("only one")]
        retrieval = _make_retrieval_result(chunks)
        mock_llm.generate.return_value = _make_llm_response("Answer [1].")

        result = await generator.generate("question", retrieval, top_k=50)
        assert result.answer == "Answer [1]."

    @pytest.mark.asyncio
    async def test_zero_chunks_triggers_abstention(
        self, generator: Generator, mock_llm: MagicMock
    ) -> None:
        retrieval = _make_retrieval_result([])
        mock_llm.generate.return_value = _make_llm_response(
            "I don't have enough information in the available sources "
            "to answer this question."
        )

        result = await generator.generate("question", retrieval)

        assert result.is_abstention is True
        assert result.confidence_score == 0.0
