from __future__ import annotations

from rag.models.document import Chunk, ChunkMetadata
from rag.models.generation import Citation
from rag.models.retrieval import RetrievalResult, ScoredChunk
from rag.models.verification import EntailmentResult, VerificationReport
from rag.verification.abstention import AbstentionDecider


def _make_scored_chunk(
    score: float, method: str = "vector"
) -> ScoredChunk:
    chunk = Chunk(
        document_id="doc-1",
        text="some text",
        metadata=ChunkMetadata(source_file="test.txt"),
    )
    return ScoredChunk(chunk=chunk, score=score, retrieval_method=method)


def _make_retrieval_result(
    scores: list[float],
    method: str = "vector",
) -> RetrievalResult:
    chunks = tuple(_make_scored_chunk(s, method) for s in scores)
    return RetrievalResult(
        query="test",
        rewritten_query=None,
        scored_chunks=chunks,
        retrieval_time_ms=10.0,
    )


def _make_citation(chunk_id: str = "c1") -> Citation:
    return Citation(
        chunk_id=chunk_id,
        source_file="test.txt",
        text_snippet="snippet",
        sentence_index=0,
    )


def _make_verification_report(
    faithful: bool, score: float
) -> VerificationReport:
    return VerificationReport(
        sentence_results=(
            EntailmentResult(
                sentence="test",
                label="entailment" if faithful else "contradiction",
                confidence=score,
                supporting_chunk_id="c1",
            ),
        ),
        overall_faithful=faithful,
        faithfulness_score=score,
    )


class TestAbstentionDecider:
    def test_no_retrieval_results(self) -> None:
        decider = AbstentionDecider()
        result = _make_retrieval_result([])

        decision = decider.decide(result, answer_text="Some answer [1].")

        assert decision.should_abstain is True
        assert "No relevant documents" in decision.reason

    def test_low_retrieval_scores(self) -> None:
        decider = AbstentionDecider(retrieval_score_threshold=0.3)
        result = _make_retrieval_result([0.1, 0.2, 0.15])

        decision = decider.decide(result, answer_text="answer")

        assert decision.should_abstain is True
        assert "low relevance" in decision.reason

    def test_good_retrieval_scores_pass(self) -> None:
        decider = AbstentionDecider()
        result = _make_retrieval_result([0.8, 0.7, 0.6])
        citations = [_make_citation()]

        decision = decider.decide(
            result,
            validated_citations=citations,
            answer_text="Good answer [1].",
        )

        assert decision.should_abstain is False

    def test_low_reranker_scores(self) -> None:
        decider = AbstentionDecider(reranker_score_threshold=0.5)
        result = _make_retrieval_result([0.3, 0.2], method="reranked")

        decision = decider.decide(result, answer_text="answer")

        assert decision.should_abstain is True
        assert "Reranked" in decision.reason

    def test_reranker_threshold_only_applies_to_reranked(self) -> None:
        decider = AbstentionDecider(
            retrieval_score_threshold=0.1,
            reranker_score_threshold=0.5,
        )
        # Vector results with score 0.35 -- above retrieval threshold
        result = _make_retrieval_result([0.35], method="vector")
        citations = [_make_citation()]

        decision = decider.decide(
            result,
            validated_citations=citations,
            answer_text="answer [1].",
        )

        # Should pass -- reranker threshold doesn't apply to vector results
        assert decision.should_abstain is False

    def test_failed_faithfulness(self) -> None:
        decider = AbstentionDecider()
        result = _make_retrieval_result([0.8])
        report = _make_verification_report(faithful=False, score=0.3)

        decision = decider.decide(
            result,
            verification_report=report,
            answer_text="answer",
        )

        assert decision.should_abstain is True
        assert "faithfulness" in decision.reason

    def test_passed_faithfulness(self) -> None:
        decider = AbstentionDecider()
        result = _make_retrieval_result([0.8])
        report = _make_verification_report(faithful=True, score=0.95)
        citations = [_make_citation()]

        decision = decider.decide(
            result,
            verification_report=report,
            validated_citations=citations,
            answer_text="Good answer [1].",
        )

        assert decision.should_abstain is False

    def test_no_verification_report_skips_check(self) -> None:
        decider = AbstentionDecider()
        result = _make_retrieval_result([0.8])
        citations = [_make_citation()]

        decision = decider.decide(
            result,
            verification_report=None,
            validated_citations=citations,
            answer_text="answer [1].",
        )

        assert decision.should_abstain is False

    def test_all_citations_stripped(self) -> None:
        decider = AbstentionDecider()
        result = _make_retrieval_result([0.8])

        decision = decider.decide(
            result,
            validated_citations=[],  # all stripped
            answer_text="An answer with citation [1].",
        )

        assert decision.should_abstain is True
        assert "citations" in decision.reason.lower()

    def test_no_citations_but_no_markers_in_answer(self) -> None:
        decider = AbstentionDecider()
        result = _make_retrieval_result([0.8])

        # Answer has no citation markers, so empty validated_citations is fine
        decision = decider.decide(
            result,
            validated_citations=[],
            answer_text="A plain answer with no citation markers.",
        )

        assert decision.should_abstain is False

    def test_empty_answer(self) -> None:
        decider = AbstentionDecider()
        result = _make_retrieval_result([0.8])

        decision = decider.decide(result, answer_text="")

        assert decision.should_abstain is True
        assert "empty" in decision.reason.lower()

    def test_whitespace_answer(self) -> None:
        decider = AbstentionDecider()
        result = _make_retrieval_result([0.8])

        decision = decider.decide(result, answer_text="   ")

        assert decision.should_abstain is True

    def test_signal_priority_retrieval_first(self) -> None:
        """Retrieval signals are checked before verification."""
        decider = AbstentionDecider()
        result = _make_retrieval_result([])  # empty retrieval
        report = _make_verification_report(faithful=True, score=0.95)

        decision = decider.decide(
            result,
            verification_report=report,
            answer_text="answer",
        )

        assert decision.should_abstain is True
        assert "No relevant documents" in decision.reason
