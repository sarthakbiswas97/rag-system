from __future__ import annotations

import logging
import re
import time
from collections.abc import Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rag.models.generation import Citation
from rag.models.retrieval import ScoredChunk
from rag.models.verification import EntailmentResult, VerificationReport

logger = logging.getLogger(__name__)

DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

_LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}


class EntailmentChecker:
    def __init__(self, model_name: str = DEFAULT_NLI_MODEL) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(self._device)
        self._model.eval()

        # Read label mapping from model config
        label2id = self._model.config.label2id
        self._entailment_id = label2id.get("entailment", 0)

        logger.info(
            "EntailmentChecker initialized",
            extra={"model": model_name, "device": device},
        )

    def check_sentence(
        self,
        premise: str,
        hypothesis: str,
    ) -> tuple[str, float]:
        """Check if premise entails hypothesis. Returns (label, confidence)."""
        inputs = self._tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        predicted_id = probs.argmax().item()
        label = _LABEL_MAP.get(predicted_id, "neutral")
        entailment_score = float(probs[self._entailment_id])

        return label, entailment_score

    def verify_answer(
        self,
        answer: str,
        scored_chunks: Sequence[ScoredChunk],
        citations: Sequence[Citation],
        faithfulness_threshold: float = 0.7,
    ) -> VerificationReport:
        if not answer.strip():
            return VerificationReport(
                sentence_results=(),
                overall_faithful=False,
                faithfulness_score=0.0,
            )

        start = time.perf_counter()
        sentences = _split_sentences(answer)

        if not sentences:
            return VerificationReport(
                sentence_results=(),
                overall_faithful=False,
                faithfulness_score=0.0,
            )

        # Build citation index: sentence_index -> chunk text
        citation_map: dict[int, list[str]] = {}
        for c in citations:
            chunk_texts = citation_map.setdefault(c.sentence_index, [])
            # Find the full chunk text from scored_chunks
            for sc in scored_chunks:
                if sc.chunk.chunk_id == c.chunk_id:
                    chunk_texts.append(sc.chunk.text)
                    break

        # All chunk texts concatenated as fallback context
        all_context = " ".join(sc.chunk.text for sc in scored_chunks[:5])

        results: list[EntailmentResult] = []
        entailed_count = 0

        for sent_idx, sentence in enumerate(sentences):
            # Use cited chunk if available, otherwise check against all context
            contexts = citation_map.get(sent_idx)
            if contexts:
                premise = " ".join(contexts)
                supporting_chunk_id = _find_supporting_chunk_id(
                    sent_idx, citations
                )
            else:
                premise = all_context
                supporting_chunk_id = None

            label, entailment_score = self.check_sentence(premise, sentence)

            if entailment_score >= faithfulness_threshold:
                entailed_count += 1

            results.append(
                EntailmentResult(
                    sentence=sentence,
                    label=label,
                    confidence=entailment_score,
                    supporting_chunk_id=supporting_chunk_id,
                )
            )

        faithfulness_score = (
            entailed_count / len(sentences) if sentences else 0.0
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Answer verification complete",
            extra={
                "sentences": len(sentences),
                "entailed": entailed_count,
                "faithfulness_score": round(faithfulness_score, 3),
                "elapsed_ms": round(elapsed_ms, 1),
            },
        )

        return VerificationReport(
            sentence_results=tuple(results),
            overall_faithful=faithfulness_score >= faithfulness_threshold,
            faithfulness_score=round(faithfulness_score, 3),
        )


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT.split(text)
    return [s.strip() for s in parts if s.strip()]


def _find_supporting_chunk_id(
    sentence_index: int,
    citations: Sequence[Citation],
) -> str | None:
    for c in citations:
        if c.sentence_index == sentence_index:
            return c.chunk_id
    return None
