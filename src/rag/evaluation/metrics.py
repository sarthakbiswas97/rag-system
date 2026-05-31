from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from rag.models.generation import Citation
from rag.models.retrieval import ScoredChunk
from rag.verification.entailment import EntailmentChecker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalSample:
    """A single evaluation example."""

    question: str
    answer: str
    contexts: tuple[str, ...]
    ground_truth: str | None = None
    citations: tuple[Citation, ...] = ()


@dataclass(frozen=True)
class SampleScores:
    """Per-sample evaluation scores."""

    faithfulness: float
    context_utilization: float
    citation_precision: float | None


@dataclass(frozen=True)
class EvalReport:
    """Aggregated evaluation results."""

    sample_scores: tuple[SampleScores, ...]
    mean_faithfulness: float
    mean_context_utilization: float
    mean_citation_precision: float | None
    total_samples: int


def compute_faithfulness(
    answer: str,
    contexts: Sequence[str],
    entailment_checker: EntailmentChecker,
    threshold: float = 0.7,
) -> float:
    """Fraction of answer sentences entailed by any context.

    Decomposes answer into sentences, checks each against all contexts.
    Returns entailed_count / total_sentences.
    """
    from rag.verification.entailment import _split_sentences

    sentences = _split_sentences(answer)
    if not sentences:
        return 0.0

    combined_context = " ".join(contexts)
    if not combined_context.strip():
        return 0.0

    entailed = 0
    for sentence in sentences:
        _label, score = entailment_checker.check_sentence(combined_context, sentence)
        if score >= threshold:
            entailed += 1

    return round(entailed / len(sentences), 3)


def compute_context_utilization(
    answer: str,
    contexts: Sequence[str],
    entailment_checker: EntailmentChecker,
    threshold: float = 0.5,
) -> float:
    """Fraction of provided contexts that contributed to the answer.

    For each context, check if any answer sentence is entailed by it.
    Returns used_contexts / total_contexts.
    """
    from rag.verification.entailment import _split_sentences

    if not contexts:
        return 0.0

    sentences = _split_sentences(answer)
    if not sentences:
        return 0.0

    used = 0
    for context in contexts:
        if not context.strip():
            continue
        for sentence in sentences:
            _label, score = entailment_checker.check_sentence(context, sentence)
            if score >= threshold:
                used += 1
                break

    return round(used / len(contexts), 3)


def compute_citation_precision(
    citations: Sequence[Citation],
    contexts: Sequence[str],
    answer: str,
    entailment_checker: EntailmentChecker,
    scored_chunks: Sequence[ScoredChunk] = (),
    threshold: float = 0.5,
) -> float | None:
    """Fraction of citations that are actually supported by their source.

    Returns None if no citations to evaluate.
    """
    from rag.verification.entailment import _split_sentences

    if not citations:
        return None

    sentences = _split_sentences(answer)
    chunk_by_id = {sc.chunk.chunk_id: sc.chunk.text for sc in scored_chunks}

    supported = 0
    for citation in citations:
        chunk_text = chunk_by_id.get(citation.chunk_id)
        if not chunk_text:
            continue

        sent_idx = citation.sentence_index
        if 0 <= sent_idx < len(sentences):
            _label, score = entailment_checker.check_sentence(
                chunk_text, sentences[sent_idx]
            )
            if score >= threshold:
                supported += 1

    return round(supported / len(citations), 3)


class RAGEvaluator:
    def __init__(self, entailment_checker: EntailmentChecker) -> None:
        self._checker = entailment_checker

    def evaluate_sample(self, sample: EvalSample) -> SampleScores:
        faithfulness = compute_faithfulness(
            sample.answer, sample.contexts, self._checker
        )
        utilization = compute_context_utilization(
            sample.answer, sample.contexts, self._checker
        )
        citation_precision = compute_citation_precision(
            sample.citations, sample.contexts, sample.answer, self._checker
        )

        return SampleScores(
            faithfulness=faithfulness,
            context_utilization=utilization,
            citation_precision=citation_precision,
        )

    def evaluate(self, samples: Sequence[EvalSample]) -> EvalReport:
        if not samples:
            return EvalReport(
                sample_scores=(),
                mean_faithfulness=0.0,
                mean_context_utilization=0.0,
                mean_citation_precision=None,
                total_samples=0,
            )

        scores = tuple(self.evaluate_sample(s) for s in samples)

        mean_faith = sum(s.faithfulness for s in scores) / len(scores)
        mean_util = sum(s.context_utilization for s in scores) / len(scores)

        cp_scores = [
            s.citation_precision for s in scores if s.citation_precision is not None
        ]
        mean_cp = sum(cp_scores) / len(cp_scores) if cp_scores else None

        report = EvalReport(
            sample_scores=scores,
            mean_faithfulness=round(mean_faith, 3),
            mean_context_utilization=round(mean_util, 3),
            mean_citation_precision=round(mean_cp, 3) if mean_cp is not None else None,
            total_samples=len(scores),
        )

        logger.info(
            "Evaluation complete",
            extra={
                "samples": len(scores),
                "mean_faithfulness": report.mean_faithfulness,
                "mean_context_utilization": report.mean_context_utilization,
                "mean_citation_precision": report.mean_citation_precision,
            },
        )

        return report
