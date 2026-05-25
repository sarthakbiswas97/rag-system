from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from rag.generation.llm_client import LLMClient
from rag.models.retrieval import RetrievalResult, ScoredChunk

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a search query optimizer. Given a user question, generate exactly 3 "
    "alternative phrasings that would help retrieve relevant documents. Each variant "
    "should capture different aspects or use different terminology.\n\n"
    "Output ONLY a numbered list (1. 2. 3.) with no other text."
)

_VARIANT_PATTERN = re.compile(r"^\d+\.\s*(.+)$", re.MULTILINE)


async def rewrite_query(
    query: str,
    llm_client: LLMClient,
) -> tuple[str, ...]:
    """Return the original query plus LLM-generated variants."""
    if not query.strip():
        return (query,)

    try:
        response = await llm_client.generate(_SYSTEM_PROMPT, query)
        variants = _VARIANT_PATTERN.findall(response.content)

        # Deduplicate and exclude anything identical to original
        seen = {query.strip().lower()}
        unique_variants: list[str] = []
        for v in variants:
            v_stripped = v.strip()
            if v_stripped.lower() not in seen and v_stripped:
                seen.add(v_stripped.lower())
                unique_variants.append(v_stripped)

        result = (query, *unique_variants)

        logger.info(
            "Query rewriting complete",
            extra={"original": query[:100], "variants": len(result) - 1},
        )

        return result

    except Exception:
        logger.warning("Query rewriting failed, using original query", exc_info=True)
        return (query,)


def merge_retrieval_results(
    results: Sequence[RetrievalResult],
) -> tuple[ScoredChunk, ...]:
    """Merge multiple retrieval results, keeping the highest score per chunk."""
    if not results:
        return ()

    best_by_chunk_id: dict[str, ScoredChunk] = {}
    for result in results:
        for sc in result.scored_chunks:
            existing = best_by_chunk_id.get(sc.chunk.chunk_id)
            if existing is None or sc.score > existing.score:
                best_by_chunk_id[sc.chunk.chunk_id] = sc

    merged = sorted(best_by_chunk_id.values(), key=lambda sc: sc.score, reverse=True)
    return tuple(merged)
