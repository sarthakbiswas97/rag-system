from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from rag.models.document import Chunk, ChunkMetadata
from rag.models.generation import LLMResponse
from rag.models.retrieval import RetrievalResult, ScoredChunk
from rag.retrieval.query_rewriter import merge_retrieval_results, rewrite_query


def _make_llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        prompt_tokens=10,
        completion_tokens=20,
        model="gpt-4o-mini",
        elapsed_ms=100.0,
    )


def _make_scored_chunk(
    chunk_id: str, text: str, score: float
) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        text=text,
        metadata=ChunkMetadata(source_file="test.txt"),
    )
    return ScoredChunk(chunk=chunk, score=score, retrieval_method="vector")


@pytest.fixture()
def mock_llm_client() -> MagicMock:
    client = MagicMock()
    client.generate = AsyncMock()
    return client


class TestRewriteQuery:
    async def test_returns_original_plus_variants(
        self, mock_llm_client: MagicMock
    ) -> None:
        mock_llm_client.generate.return_value = _make_llm_response(
            "1. What is the capital city of France?\n"
            "2. Which city serves as France's capital?\n"
            "3. Capital of France\n"
        )
        result = await rewrite_query("capital of France?", mock_llm_client)

        assert result[0] == "capital of France?"
        assert len(result) == 4  # original + 3 variants

    async def test_deduplicates_variants(
        self, mock_llm_client: MagicMock
    ) -> None:
        mock_llm_client.generate.return_value = _make_llm_response(
            "1. Capital of France?\n"
            "2. capital of france?\n"
            "3. What is the capital of France?\n"
        )
        result = await rewrite_query("capital of France?", mock_llm_client)

        # "Capital of France?" matches original (case-insensitive) -> excluded
        # "capital of france?" also matches -> excluded
        assert result[0] == "capital of France?"
        assert len(result) == 2  # original + 1 unique variant

    async def test_empty_query_returns_original(
        self, mock_llm_client: MagicMock
    ) -> None:
        result = await rewrite_query("", mock_llm_client)

        assert result == ("",)
        mock_llm_client.generate.assert_not_called()

    async def test_blank_query_returns_original(
        self, mock_llm_client: MagicMock
    ) -> None:
        result = await rewrite_query("   ", mock_llm_client)

        assert result == ("   ",)
        mock_llm_client.generate.assert_not_called()

    async def test_llm_failure_returns_original(
        self, mock_llm_client: MagicMock
    ) -> None:
        mock_llm_client.generate.side_effect = RuntimeError("API down")

        result = await rewrite_query("test query", mock_llm_client)

        assert result == ("test query",)

    async def test_malformed_llm_output(
        self, mock_llm_client: MagicMock
    ) -> None:
        mock_llm_client.generate.return_value = _make_llm_response(
            "Here are some alternatives:\n- variant one\n- variant two"
        )
        result = await rewrite_query("test query", mock_llm_client)

        # No numbered list found -> just original
        assert result == ("test query",)

    async def test_strips_whitespace_from_variants(
        self, mock_llm_client: MagicMock
    ) -> None:
        mock_llm_client.generate.return_value = _make_llm_response(
            "1.   padded variant   \n"
            "2. clean variant\n"
            "3. another one\n"
        )
        result = await rewrite_query("query", mock_llm_client)

        for v in result[1:]:
            assert v == v.strip()


class TestMergeRetrievalResults:
    def test_merges_and_deduplicates(self) -> None:
        result1 = RetrievalResult(
            query="q1",
            rewritten_query=None,
            scored_chunks=(
                _make_scored_chunk("c1", "text1", 0.8),
                _make_scored_chunk("c2", "text2", 0.6),
            ),
            retrieval_time_ms=10.0,
        )
        result2 = RetrievalResult(
            query="q2",
            rewritten_query=None,
            scored_chunks=(
                _make_scored_chunk("c1", "text1", 0.9),  # same chunk, higher score
                _make_scored_chunk("c3", "text3", 0.7),
            ),
            retrieval_time_ms=10.0,
        )

        merged = merge_retrieval_results([result1, result2])

        assert len(merged) == 3
        # c1 should have score 0.9 (higher)
        c1 = next(sc for sc in merged if sc.chunk.chunk_id == "c1")
        assert c1.score == 0.9

    def test_sorted_descending_by_score(self) -> None:
        result = RetrievalResult(
            query="q",
            rewritten_query=None,
            scored_chunks=(
                _make_scored_chunk("c1", "t1", 0.3),
                _make_scored_chunk("c2", "t2", 0.9),
                _make_scored_chunk("c3", "t3", 0.6),
            ),
            retrieval_time_ms=10.0,
        )

        merged = merge_retrieval_results([result])

        scores = [sc.score for sc in merged]
        assert scores == [0.9, 0.6, 0.3]

    def test_empty_results(self) -> None:
        assert merge_retrieval_results([]) == ()

    def test_single_result_passthrough(self) -> None:
        result = RetrievalResult(
            query="q",
            rewritten_query=None,
            scored_chunks=(
                _make_scored_chunk("c1", "t1", 0.8),
            ),
            retrieval_time_ms=10.0,
        )

        merged = merge_retrieval_results([result])

        assert len(merged) == 1
        assert merged[0].score == 0.8
