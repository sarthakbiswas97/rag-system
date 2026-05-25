from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from rag.models.generation import Citation
from rag.models.retrieval import ScoredChunk
from rag.verification.entailment import EntailmentChecker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CitationVerification:
    citation: Citation
    is_supported: bool
    entailment_score: float


class CitationValidator:
    def __init__(
        self,
        entailment_checker: EntailmentChecker,
        support_threshold: float = 0.5,
    ) -> None:
        self._checker = entailment_checker
        self._support_threshold = support_threshold

    def validate(
        self,
        answer: str,
        citations: Sequence[Citation],
        scored_chunks: Sequence[ScoredChunk],
    ) -> tuple[CitationVerification, ...]:
        if not citations or not scored_chunks:
            return ()

        # Build chunk lookup
        chunk_by_id: dict[str, str] = {
            sc.chunk.chunk_id: sc.chunk.text for sc in scored_chunks
        }

        # Split answer into sentences for context
        sentences = _extract_sentences(answer)

        results: list[CitationVerification] = []

        for citation in citations:
            chunk_text = chunk_by_id.get(citation.chunk_id)
            if chunk_text is None:
                # Citation references unknown chunk -> unsupported
                results.append(
                    CitationVerification(
                        citation=citation,
                        is_supported=False,
                        entailment_score=0.0,
                    )
                )
                continue

            # Get the sentence this citation is attached to
            sentence = _get_sentence(sentences, citation.sentence_index)
            if not sentence:
                results.append(
                    CitationVerification(
                        citation=citation,
                        is_supported=False,
                        entailment_score=0.0,
                    )
                )
                continue

            # Check: does the chunk text entail the sentence?
            _label, entailment_score = self._checker.check_sentence(
                chunk_text, sentence
            )

            is_supported = entailment_score >= self._support_threshold

            results.append(
                CitationVerification(
                    citation=citation,
                    is_supported=is_supported,
                    entailment_score=entailment_score,
                )
            )

        supported = sum(1 for r in results if r.is_supported)
        logger.info(
            "Citation validation complete",
            extra={
                "total": len(results),
                "supported": supported,
                "unsupported": len(results) - supported,
            },
        )

        return tuple(results)


def filter_supported_citations(
    verifications: Sequence[CitationVerification],
) -> tuple[Citation, ...]:
    """Return only citations that passed verification."""
    return tuple(v.citation for v in verifications if v.is_supported)


def _extract_sentences(text: str) -> list[str]:
    import re

    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if s.strip()]


def _get_sentence(sentences: list[str], index: int) -> str | None:
    if 0 <= index < len(sentences):
        return sentences[index]
    return None
