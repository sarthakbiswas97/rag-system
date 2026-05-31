from __future__ import annotations

from unittest.mock import MagicMock

from rag.evaluation.metrics import (
    EvalSample,
    RAGEvaluator,
    compute_citation_precision,
    compute_context_utilization,
    compute_faithfulness,
)
from rag.models.document import Chunk, ChunkMetadata
from rag.models.generation import Citation
from rag.models.retrieval import ScoredChunk


def _make_checker(results: list[tuple[str, float]]) -> MagicMock:
    """Mock entailment checker returning given results in order."""
    checker = MagicMock()
    call_count = 0

    def mock_check(premise: str, hypothesis: str) -> tuple[str, float]:
        nonlocal call_count
        r = results[call_count] if call_count < len(results) else results[-1]
        call_count += 1
        return r

    checker.check_sentence = mock_check
    return checker


def _make_scored_chunk(chunk_id: str, text: str) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        text=text,
        metadata=ChunkMetadata(source_file="test.txt"),
    )
    return ScoredChunk(chunk=chunk, score=0.9, retrieval_method="vector")


class TestComputeFaithfulness:
    def test_all_entailed(self) -> None:
        checker = _make_checker(
            [
                ("entailment", 0.95),
                ("entailment", 0.9),
            ]
        )
        score = compute_faithfulness(
            "Sentence one. Sentence two.",
            ["context text"],
            checker,
        )
        assert score == 1.0

    def test_partial_entailment(self) -> None:
        checker = _make_checker(
            [
                ("entailment", 0.95),
                ("contradiction", 0.1),
            ]
        )
        score = compute_faithfulness(
            "Supported claim. Unsupported claim.",
            ["context"],
            checker,
        )
        assert score == 0.5

    def test_empty_answer(self) -> None:
        checker = _make_checker([("entailment", 0.95)])
        assert compute_faithfulness("", ["context"], checker) == 0.0

    def test_empty_contexts(self) -> None:
        checker = _make_checker([("entailment", 0.95)])
        assert compute_faithfulness("answer.", [], checker) == 0.0


class TestComputeContextUtilization:
    def test_all_contexts_used(self) -> None:
        checker = _make_checker(
            [
                ("entailment", 0.8),
                ("entailment", 0.8),
            ]
        )
        score = compute_context_utilization(
            "Answer sentence.",
            ["context1", "context2"],
            checker,
        )
        assert score == 1.0

    def test_partial_utilization(self) -> None:
        checker = _make_checker(
            [
                ("entailment", 0.8),  # context1 -> sentence: entailed
                ("neutral", 0.1),  # context2 -> sentence: not entailed
            ]
        )
        score = compute_context_utilization(
            "Answer sentence.",
            ["relevant context", "irrelevant context"],
            checker,
        )
        assert score == 0.5

    def test_empty_contexts(self) -> None:
        checker = _make_checker([("entailment", 0.95)])
        assert compute_context_utilization("answer.", [], checker) == 0.0

    def test_empty_answer(self) -> None:
        checker = _make_checker([("entailment", 0.95)])
        assert compute_context_utilization("", ["ctx"], checker) == 0.0


class TestComputeCitationPrecision:
    def test_all_supported(self) -> None:
        checker = _make_checker([("entailment", 0.9)])
        chunks = [_make_scored_chunk("c1", "Paris is the capital.")]
        citations = [
            Citation(
                chunk_id="c1",
                source_file="test.txt",
                text_snippet="snippet",
                sentence_index=0,
            )
        ]
        score = compute_citation_precision(
            citations, ["ctx"], "Paris is the capital.", checker, chunks
        )
        assert score == 1.0

    def test_no_citations_returns_none(self) -> None:
        checker = _make_checker([("entailment", 0.9)])
        assert compute_citation_precision([], [], "answer", checker) is None

    def test_unknown_chunk_id(self) -> None:
        checker = _make_checker([("entailment", 0.9)])
        citations = [
            Citation(
                chunk_id="unknown",
                source_file="test.txt",
                text_snippet="snippet",
                sentence_index=0,
            )
        ]
        score = compute_citation_precision(citations, ["ctx"], "answer.", checker, [])
        assert score == 0.0


class TestRAGEvaluator:
    def test_evaluate_single_sample(self) -> None:
        checker = _make_checker(
            [
                # faithfulness: 1 sentence, entailed
                ("entailment", 0.95),
                # context_utilization: 1 context checked
                ("entailment", 0.8),
            ]
        )
        evaluator = RAGEvaluator(entailment_checker=checker)
        sample = EvalSample(
            question="What is X?",
            answer="X is Y.",
            contexts=("X is Y and Z.",),
        )

        report = evaluator.evaluate([sample])

        assert report.total_samples == 1
        assert report.mean_faithfulness == 1.0
        assert report.mean_context_utilization == 1.0
        assert report.mean_citation_precision is None

    def test_evaluate_empty(self) -> None:
        checker = _make_checker([("entailment", 0.95)])
        evaluator = RAGEvaluator(entailment_checker=checker)

        report = evaluator.evaluate([])

        assert report.total_samples == 0
        assert report.mean_faithfulness == 0.0

    def test_evaluate_multiple_samples(self) -> None:
        checker = _make_checker(
            [
                # Sample 1 faithfulness
                ("entailment", 0.95),
                # Sample 1 context utilization
                ("entailment", 0.8),
                # Sample 2 faithfulness
                ("neutral", 0.3),
                # Sample 2 context utilization
                ("neutral", 0.2),
            ]
        )
        evaluator = RAGEvaluator(entailment_checker=checker)
        samples = [
            EvalSample(
                question="Q1",
                answer="Good answer.",
                contexts=("context1",),
            ),
            EvalSample(
                question="Q2",
                answer="Bad answer.",
                contexts=("context2",),
            ),
        ]

        report = evaluator.evaluate(samples)

        assert report.total_samples == 2
        assert report.mean_faithfulness == 0.5  # (1.0 + 0.0) / 2
