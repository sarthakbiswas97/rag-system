from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from rag.generation.llm_client import LLMClient
from rag.generation.prompt_builder import build_rag_prompt
from rag.models.generation import Citation, GenerationResponse
from rag.models.retrieval import RetrievalResult, ScoredChunk

logger = logging.getLogger(__name__)

SNIPPET_MAX_LEN = 200

ABSTENTION_PHRASES = (
    "i don't have enough information",
    "not enough information",
    "cannot answer",
    "no relevant sources",
    "no sources available",
)

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT.split(text)
    return [s.strip() for s in parts if s.strip()]


def parse_citations(
    text: str,
    scored_chunks: Sequence[ScoredChunk],
) -> tuple[Citation, ...]:
    if not text or not scored_chunks:
        return ()

    sentences = _split_sentences(text)
    seen: set[tuple[str, int]] = set()
    citations: list[Citation] = []

    for sent_idx, sentence in enumerate(sentences):
        indices = _CITATION_PATTERN.findall(sentence)
        unique_indices = sorted(set(int(i) for i in indices))

        for idx in unique_indices:
            if idx < 1 or idx > len(scored_chunks):
                continue

            chunk = scored_chunks[idx - 1].chunk
            key = (chunk.chunk_id, sent_idx)
            if key in seen:
                continue
            seen.add(key)

            snippet = chunk.text[:SNIPPET_MAX_LEN]
            citations.append(
                Citation(
                    chunk_id=chunk.chunk_id,
                    source_file=chunk.metadata.source_file,
                    text_snippet=snippet,
                    sentence_index=sent_idx,
                )
            )

    return tuple(citations)


def detect_abstention(text: str, has_citations: bool) -> bool:
    if not text.strip():
        return True

    if has_citations:
        return False

    lowered = text.lower()
    return any(phrase in lowered for phrase in ABSTENTION_PHRASES)


def compute_confidence(scored_chunks: Sequence[ScoredChunk]) -> float:
    if not scored_chunks:
        return 0.0

    mean = sum(sc.score for sc in scored_chunks) / len(scored_chunks)
    return max(0.0, min(1.0, mean))


class Generator:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    async def generate(
        self,
        question: str,
        retrieval_result: RetrievalResult,
        top_k: int = 5,
    ) -> GenerationResponse:
        top_chunks = retrieval_result.scored_chunks[:top_k]

        system_prompt, user_prompt = build_rag_prompt(question, top_chunks)
        llm_response = await self._llm_client.generate(system_prompt, user_prompt)

        citations = parse_citations(llm_response.content, top_chunks)
        is_abstention = detect_abstention(
            llm_response.content, has_citations=len(citations) > 0
        )
        confidence = compute_confidence(top_chunks)

        logger.info(
            "Generation complete",
            extra={
                "citations": len(citations),
                "is_abstention": is_abstention,
                "confidence": round(confidence, 3),
            },
        )

        return GenerationResponse(
            answer=llm_response.content,
            citations=citations,
            is_abstention=is_abstention,
            confidence_score=confidence,
            retrieval_result=retrieval_result,
            generation_time_ms=llm_response.elapsed_ms,
        )
