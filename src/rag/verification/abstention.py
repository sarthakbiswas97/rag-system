from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from rag.models.generation import Citation
from rag.models.retrieval import RetrievalResult
from rag.models.verification import VerificationReport

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AbstentionDecision:
    should_abstain: bool
    reason: str


class AbstentionDecider:
    def __init__(
        self,
        retrieval_score_threshold: float = 0.3,
        reranker_score_threshold: float = 0.5,
        faithfulness_threshold: float = 0.7,
    ) -> None:
        self._retrieval_threshold = retrieval_score_threshold
        self._reranker_threshold = reranker_score_threshold
        self._faithfulness_threshold = faithfulness_threshold

    def decide(
        self,
        retrieval_result: RetrievalResult,
        verification_report: VerificationReport | None = None,
        validated_citations: Sequence[Citation] = (),
        answer_text: str = "",
    ) -> AbstentionDecision:
        # Signal 1: No retrieval results at all
        if not retrieval_result.scored_chunks:
            return AbstentionDecision(
                should_abstain=True,
                reason="No relevant documents found.",
            )

        # Signal 2: All retrieval scores below threshold
        max_score = max(
            sc.score for sc in retrieval_result.scored_chunks
        )
        if max_score < self._retrieval_threshold:
            return AbstentionDecision(
                should_abstain=True,
                reason="Retrieved documents have low relevance scores.",
            )

        # Signal 3: Reranked scores below threshold (if reranking was used)
        is_reranked = any(
            sc.retrieval_method == "reranked"
            for sc in retrieval_result.scored_chunks
        )
        if is_reranked and max_score < self._reranker_threshold:
            return AbstentionDecision(
                should_abstain=True,
                reason="Reranked documents have low relevance scores.",
            )

        # Signal 4: Faithfulness score below threshold
        if (
            verification_report is not None
            and verification_report.faithfulness_score < self._faithfulness_threshold
        ):
            return AbstentionDecision(
                should_abstain=True,
                reason=(
                    f"Answer failed faithfulness check "
                    f"(score: {verification_report.faithfulness_score:.2f}, "
                    f"threshold: {self._faithfulness_threshold:.2f})."
                ),
            )

        # Signal 5: All citations stripped by validator
        if answer_text.strip() and not validated_citations:
            # Had an answer but no citations survived validation
            has_citation_markers = "[" in answer_text and "]" in answer_text
            if has_citation_markers:
                return AbstentionDecision(
                    should_abstain=True,
                    reason="No citations could be verified against sources.",
                )

        # Signal 6: Empty or whitespace-only answer
        if not answer_text.strip():
            return AbstentionDecision(
                should_abstain=True,
                reason="Generated answer is empty.",
            )

        logger.debug(
            "Abstention check passed",
            extra={
                "max_retrieval_score": round(max_score, 3),
                "faithfulness": (
                    verification_report.faithfulness_score
                    if verification_report
                    else None
                ),
                "validated_citations": len(validated_citations),
            },
        )

        return AbstentionDecision(
            should_abstain=False,
            reason="",
        )
