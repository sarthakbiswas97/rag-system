from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

from rag.models.generation import Citation, GenerationResponse
from rag.models.retrieval import ScoredChunk
from rag.models.verification import VerificationReport
from rag.verification.abstention import AbstentionDecider, AbstentionDecision
from rag.verification.citation_validator import (
    CitationValidator,
    CitationVerification,
    filter_supported_citations,
)
from rag.verification.entailment import EntailmentChecker

logger = logging.getLogger(__name__)

ABSTENTION_ANSWER = (
    "I don't have enough information in the provided sources "
    "to answer this question reliably."
)


@dataclass(frozen=True)
class VerificationResult:
    answer: str
    citations: tuple[Citation, ...]
    is_abstention: bool
    confidence_score: float
    verification_report: VerificationReport | None
    citation_verifications: tuple[CitationVerification, ...] | None
    abstention_decision: AbstentionDecision | None
    verification_time_ms: float


class VerificationPipeline:
    def __init__(
        self,
        entailment_checker: EntailmentChecker,
        citation_validator: CitationValidator,
        abstention_decider: AbstentionDecider,
    ) -> None:
        self._entailment_checker = entailment_checker
        self._citation_validator = citation_validator
        self._abstention_decider = abstention_decider

    def verify(
        self,
        generation_response: GenerationResponse,
        top_chunks: Sequence[ScoredChunk],
    ) -> VerificationResult:
        start = time.perf_counter()

        retrieval_result = generation_response.retrieval_result

        # Step 1: NLI entailment check
        verification_report = self._entailment_checker.verify_answer(
            generation_response.answer,
            top_chunks,
            generation_response.citations,
        )

        # Step 2: Citation validation
        citation_verifications = self._citation_validator.validate(
            generation_response.answer,
            generation_response.citations,
            top_chunks,
        )
        validated_citations = filter_supported_citations(citation_verifications)

        # Step 3: Abstention decision
        abstention_decision = self._abstention_decider.decide(
            retrieval_result=retrieval_result,
            verification_report=verification_report,
            validated_citations=validated_citations,
            answer_text=generation_response.answer,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        if abstention_decision.should_abstain:
            logger.info(
                "Verification triggered abstention",
                extra={
                    "reason": abstention_decision.reason,
                    "faithfulness_score": verification_report.faithfulness_score,
                    "elapsed_ms": round(elapsed_ms, 1),
                },
            )
            return VerificationResult(
                answer=ABSTENTION_ANSWER,
                citations=(),
                is_abstention=True,
                confidence_score=0.0,
                verification_report=verification_report,
                citation_verifications=citation_verifications,
                abstention_decision=abstention_decision,
                verification_time_ms=round(elapsed_ms, 1),
            )

        logger.info(
            "Verification passed",
            extra={
                "faithfulness_score": verification_report.faithfulness_score,
                "citations_kept": len(validated_citations),
                "citations_total": len(generation_response.citations),
                "elapsed_ms": round(elapsed_ms, 1),
            },
        )

        return VerificationResult(
            answer=generation_response.answer,
            citations=validated_citations,
            is_abstention=False,
            confidence_score=generation_response.confidence_score,
            verification_report=verification_report,
            citation_verifications=citation_verifications,
            abstention_decision=abstention_decision,
            verification_time_ms=round(elapsed_ms, 1),
        )
