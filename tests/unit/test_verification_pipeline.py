from __future__ import annotations

from unittest.mock import MagicMock

from rag.models.document import Chunk, ChunkMetadata
from rag.models.generation import Citation, GenerationResponse
from rag.models.retrieval import RetrievalResult, ScoredChunk
from rag.models.verification import EntailmentResult, VerificationReport
from rag.verification.abstention import AbstentionDecider, AbstentionDecision
from rag.verification.citation_validator import (
    CitationValidator,
    CitationVerification,
)
from rag.verification.pipeline import ABSTENTION_ANSWER, VerificationPipeline


def _make_scored_chunk(chunk_id: str, text: str, score: float) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        text=text,
        metadata=ChunkMetadata(source_file="test.txt"),
    )
    return ScoredChunk(chunk=chunk, score=score, retrieval_method="vector")


def _make_citation(chunk_id: str, sent_idx: int = 0) -> Citation:
    return Citation(
        chunk_id=chunk_id,
        source_file="test.txt",
        text_snippet="snippet",
        sentence_index=sent_idx,
    )


def _make_generation_response(
    answer: str,
    citations: tuple[Citation, ...] = (),
) -> GenerationResponse:
    retrieval = RetrievalResult(
        query="test",
        rewritten_query=None,
        scored_chunks=(_make_scored_chunk("c1", "context", 0.9),),
        retrieval_time_ms=10.0,
    )
    return GenerationResponse(
        answer=answer,
        citations=citations,
        is_abstention=False,
        confidence_score=0.9,
        retrieval_result=retrieval,
        generation_time_ms=100.0,
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


def _make_pipeline(
    faithful: bool = True,
    faithfulness_score: float = 0.95,
    citation_supported: bool = True,
    should_abstain: bool = False,
    abstention_reason: str = "",
) -> VerificationPipeline:
    mock_checker = MagicMock()
    mock_checker.verify_answer.return_value = _make_verification_report(
        faithful, faithfulness_score
    )

    mock_validator = MagicMock(spec=CitationValidator)
    citation = _make_citation("c1")
    mock_validator.validate.return_value = (
        CitationVerification(
            citation=citation,
            is_supported=citation_supported,
            entailment_score=0.9 if citation_supported else 0.1,
        ),
    )

    mock_decider = MagicMock(spec=AbstentionDecider)
    mock_decider.decide.return_value = AbstentionDecision(
        should_abstain=should_abstain,
        reason=abstention_reason,
    )

    return VerificationPipeline(
        entailment_checker=mock_checker,
        citation_validator=mock_validator,
        abstention_decider=mock_decider,
    )


class TestVerificationPipeline:
    def test_passes_faithful_answer(self) -> None:
        pipeline = _make_pipeline(faithful=True)
        gen = _make_generation_response(
            "Paris is the capital [1].",
            citations=(_make_citation("c1"),),
        )
        chunks = [_make_scored_chunk("c1", "Paris is capital", 0.9)]

        result = pipeline.verify(gen, chunks)

        assert result.is_abstention is False
        assert result.answer == "Paris is the capital [1]."
        assert result.verification_report is not None
        assert result.citation_verifications is not None
        assert result.verification_time_ms >= 0

    def test_abstains_when_decided(self) -> None:
        pipeline = _make_pipeline(
            should_abstain=True,
            abstention_reason="Failed faithfulness check.",
        )
        gen = _make_generation_response("Hallucinated answer [1].")
        chunks = [_make_scored_chunk("c1", "context", 0.9)]

        result = pipeline.verify(gen, chunks)

        assert result.is_abstention is True
        assert result.answer == ABSTENTION_ANSWER
        assert result.citations == ()
        assert result.confidence_score == 0.0

    def test_passes_through_verification_metadata(self) -> None:
        pipeline = _make_pipeline(
            faithful=True,
            faithfulness_score=0.85,
        )
        gen = _make_generation_response("Answer [1].", (_make_citation("c1"),))
        chunks = [_make_scored_chunk("c1", "context", 0.9)]

        result = pipeline.verify(gen, chunks)

        assert result.verification_report.faithfulness_score == 0.85
        assert len(result.citation_verifications) == 1

    def test_filters_unsupported_citations(self) -> None:
        pipeline = _make_pipeline(citation_supported=False)
        gen = _make_generation_response("Answer [1].", (_make_citation("c1"),))
        chunks = [_make_scored_chunk("c1", "context", 0.9)]

        result = pipeline.verify(gen, chunks)

        # Citations should be filtered to only supported ones
        # Since citation_supported=False, filter_supported_citations returns ()
        # But abstention decision is mocked to False, so answer passes through
        assert result.citations == ()

    def test_preserves_confidence_when_not_abstaining(self) -> None:
        pipeline = _make_pipeline(faithful=True)
        gen = _make_generation_response("Good answer [1].", (_make_citation("c1"),))
        gen = GenerationResponse(
            answer=gen.answer,
            citations=gen.citations,
            is_abstention=False,
            confidence_score=0.87,
            retrieval_result=gen.retrieval_result,
            generation_time_ms=100.0,
        )
        chunks = [_make_scored_chunk("c1", "context", 0.9)]

        result = pipeline.verify(gen, chunks)

        assert result.confidence_score == 0.87
