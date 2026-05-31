from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag.models.document import Chunk, ChunkMetadata
from rag.models.retrieval import ScoredChunk
from rag.retrieval.reranker import Reranker


def _make_scored_chunk(text: str, score: float) -> ScoredChunk:
    chunk = Chunk(
        document_id="doc-1",
        text=text,
        metadata=ChunkMetadata(source_file="test.txt"),
    )
    return ScoredChunk(chunk=chunk, score=score, retrieval_method="vector")


@pytest.fixture()
def mock_cross_encoder() -> MagicMock:
    mock = MagicMock()
    return mock


@pytest.fixture()
def reranker(mock_cross_encoder: MagicMock) -> Reranker:
    with patch("rag.retrieval.reranker.CrossEncoder", return_value=mock_cross_encoder):
        r = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    r._model = mock_cross_encoder
    return r


class TestReranker:
    def test_reranks_by_cross_encoder_score(self, reranker: Reranker) -> None:
        chunks = [
            _make_scored_chunk("low relevance text", score=0.9),
            _make_scored_chunk("high relevance text", score=0.5),
            _make_scored_chunk("medium relevance text", score=0.7),
        ]
        # Cross-encoder says chunk 1 (high relevance) is best
        reranker._model.predict.return_value = [0.1, 0.95, 0.5]

        result = reranker.rerank("test query", chunks, top_k=3)

        assert len(result) == 3
        assert result[0].chunk.text == "high relevance text"
        assert result[0].score == 0.95
        assert result[0].retrieval_method == "reranked"

    def test_respects_top_k(self, reranker: Reranker) -> None:
        chunks = [_make_scored_chunk(f"text {i}", score=0.5) for i in range(10)]
        reranker._model.predict.return_value = list(range(10))

        result = reranker.rerank("query", chunks, top_k=3)

        assert len(result) == 3

    def test_top_k_larger_than_input(self, reranker: Reranker) -> None:
        chunks = [
            _make_scored_chunk("only chunk", score=0.8),
        ]
        reranker._model.predict.return_value = [0.9]

        result = reranker.rerank("query", chunks, top_k=5)

        assert len(result) == 1
        assert result[0].score == 0.9

    def test_empty_input_returns_empty(self, reranker: Reranker) -> None:
        result = reranker.rerank("query", [], top_k=5)

        assert result == ()
        reranker._model.predict.assert_not_called()

    def test_passes_correct_pairs_to_model(self, reranker: Reranker) -> None:
        chunks = [
            _make_scored_chunk("Paris is the capital", score=0.9),
            _make_scored_chunk("Berlin is in Germany", score=0.7),
        ]
        reranker._model.predict.return_value = [0.8, 0.3]

        reranker.rerank("What is the capital of France?", chunks, top_k=2)

        pairs = reranker._model.predict.call_args[0][0]
        assert pairs == [
            ("What is the capital of France?", "Paris is the capital"),
            ("What is the capital of France?", "Berlin is in Germany"),
        ]

    def test_preserves_chunk_data(self, reranker: Reranker) -> None:
        chunk = _make_scored_chunk("some text", score=0.5)
        reranker._model.predict.return_value = [0.75]

        result = reranker.rerank("query", [chunk], top_k=1)

        assert result[0].chunk.document_id == "doc-1"
        assert result[0].chunk.text == "some text"
        assert result[0].chunk.metadata.source_file == "test.txt"

    def test_sorted_descending_by_score(self, reranker: Reranker) -> None:
        chunks = [_make_scored_chunk(f"text {i}", score=0.5) for i in range(5)]
        reranker._model.predict.return_value = [0.1, 0.5, 0.3, 0.9, 0.7]

        result = reranker.rerank("query", chunks, top_k=5)

        scores = [sc.score for sc in result]
        assert scores == sorted(scores, reverse=True)
        assert scores == [0.9, 0.7, 0.5, 0.3, 0.1]
