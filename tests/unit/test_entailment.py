from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from rag.models.document import Chunk, ChunkMetadata
from rag.models.generation import Citation
from rag.models.retrieval import ScoredChunk
from rag.verification.entailment import (
    EntailmentChecker,
    _find_supporting_chunk_id,
    _split_sentences,
)


def _make_scored_chunk(chunk_id: str, text: str, score: float) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        text=text,
        metadata=ChunkMetadata(source_file="test.txt"),
    )
    return ScoredChunk(chunk=chunk, score=score, retrieval_method="vector")


def _make_citation(
    chunk_id: str, sentence_index: int, source: str = "test.txt"
) -> Citation:
    return Citation(
        chunk_id=chunk_id,
        source_file=source,
        text_snippet="snippet",
        sentence_index=sentence_index,
    )


def _make_mock_checker(
    check_results: list[tuple[str, float]],
) -> EntailmentChecker:
    """Create a checker with mocked check_sentence that returns given results.

    Args:
        check_results: List of (label, entailment_score) tuples, one per call.
    """
    with (
        patch("rag.verification.entailment.AutoTokenizer.from_pretrained"),
        patch(
            "rag.verification.entailment.AutoModelForSequenceClassification.from_pretrained"
        ) as mock_model_cls,
    ):
        mock_model = MagicMock()
        mock_model.config.label2id = {
            "entailment": 0,
            "neutral": 1,
            "contradiction": 2,
        }
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = None
        mock_model_cls.return_value = mock_model

        checker = EntailmentChecker.__new__(EntailmentChecker)
        checker._device = torch.device("cpu")
        checker._tokenizer = MagicMock()
        checker._model = mock_model
        checker._entailment_id = 0

    call_count = 0

    def mock_check_sentence(premise: str, hypothesis: str) -> tuple[str, float]:
        nonlocal call_count
        if call_count < len(check_results):
            result = check_results[call_count]
        else:
            result = check_results[-1]
        call_count += 1
        return result

    checker.check_sentence = mock_check_sentence  # type: ignore[method-assign]
    return checker


class TestSplitSentences:
    def test_splits_on_period(self) -> None:
        result = _split_sentences("First sentence. Second sentence.")
        assert result == ["First sentence.", "Second sentence."]

    def test_splits_on_exclamation(self) -> None:
        result = _split_sentences("Wow! That is great.")
        assert result == ["Wow!", "That is great."]

    def test_splits_on_question(self) -> None:
        result = _split_sentences("Is this good? Yes it is.")
        assert result == ["Is this good?", "Yes it is."]

    def test_empty_string(self) -> None:
        assert _split_sentences("") == []

    def test_whitespace_only(self) -> None:
        assert _split_sentences("   ") == []

    def test_single_sentence(self) -> None:
        result = _split_sentences("Just one sentence.")
        assert result == ["Just one sentence."]


class TestFindSupportingChunkId:
    def test_finds_matching_citation(self) -> None:
        citations = [
            _make_citation("c1", 0),
            _make_citation("c2", 1),
        ]
        assert _find_supporting_chunk_id(0, citations) == "c1"
        assert _find_supporting_chunk_id(1, citations) == "c2"

    def test_returns_none_when_no_match(self) -> None:
        citations = [_make_citation("c1", 0)]
        assert _find_supporting_chunk_id(5, citations) is None

    def test_empty_citations(self) -> None:
        assert _find_supporting_chunk_id(0, []) is None


class TestVerifyAnswer:
    def test_all_entailed(self) -> None:
        checker = _make_mock_checker(
            [
                ("entailment", 0.95),
                ("entailment", 0.92),
            ]
        )

        chunks = [_make_scored_chunk("c1", "Paris is France's capital.", 0.9)]
        citations = [_make_citation("c1", 0), _make_citation("c1", 1)]

        report = checker.verify_answer(
            "Paris is the capital [1]. It is in France [1].",
            chunks,
            citations,
        )

        assert report.overall_faithful is True
        assert report.faithfulness_score == 1.0
        assert len(report.sentence_results) == 2

    def test_partial_entailment(self) -> None:
        checker = _make_mock_checker(
            [
                ("entailment", 0.95),
                ("contradiction", 0.1),
            ]
        )

        chunks = [_make_scored_chunk("c1", "Paris is France's capital.", 0.9)]
        citations = [_make_citation("c1", 0)]

        report = checker.verify_answer(
            "Paris is the capital [1]. Berlin is also a capital.",
            chunks,
            citations,
        )

        assert report.faithfulness_score == 0.5
        assert report.overall_faithful is False  # 0.5 < 0.7 threshold

    def test_empty_answer(self) -> None:
        checker = _make_mock_checker([("entailment", 0.95)])

        report = checker.verify_answer("", [], [])

        assert report.overall_faithful is False
        assert report.faithfulness_score == 0.0
        assert report.sentence_results == ()

    def test_whitespace_answer(self) -> None:
        checker = _make_mock_checker([("entailment", 0.95)])

        report = checker.verify_answer("   ", [], [])

        assert report.overall_faithful is False

    def test_no_citations_uses_all_context(self) -> None:
        checker = _make_mock_checker([("entailment", 0.9)])

        chunks = [
            _make_scored_chunk("c1", "Paris is the capital.", 0.9),
            _make_scored_chunk("c2", "France is in Europe.", 0.8),
        ]

        report = checker.verify_answer(
            "Paris is in Europe.",
            chunks,
            [],  # no citations
        )

        assert len(report.sentence_results) == 1
        assert report.sentence_results[0].supporting_chunk_id is None

    def test_custom_threshold(self) -> None:
        checker = _make_mock_checker([("neutral", 0.5)])

        chunks = [_make_scored_chunk("c1", "context", 0.9)]

        report = checker.verify_answer(
            "A claim.",
            chunks,
            [],
            faithfulness_threshold=0.9,
        )

        assert report.overall_faithful is False

    def test_sentence_results_have_correct_labels(self) -> None:
        checker = _make_mock_checker(
            [
                ("entailment", 0.95),
                ("neutral", 0.3),
                ("contradiction", 0.05),
            ]
        )

        chunks = [_make_scored_chunk("c1", "context text", 0.9)]

        report = checker.verify_answer(
            "Sentence one. Sentence two. Sentence three.",
            chunks,
            [],
        )

        assert len(report.sentence_results) == 3
        assert report.sentence_results[0].label == "entailment"
        assert report.sentence_results[1].label == "neutral"
        assert report.sentence_results[2].label == "contradiction"

    def test_faithfulness_score_rounded(self) -> None:
        checker = _make_mock_checker(
            [
                ("entailment", 0.95),
                ("entailment", 0.85),
                ("neutral", 0.3),
            ]
        )

        chunks = [_make_scored_chunk("c1", "context", 0.9)]

        report = checker.verify_answer(
            "One. Two. Three.",
            chunks,
            [],
        )

        # 2 out of 3 entailed = 0.667
        assert report.faithfulness_score == 0.667
