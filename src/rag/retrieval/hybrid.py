from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from rag.models.retrieval import ScoredChunk

RRF_K = 60


def reciprocal_rank_fusion(
    *result_lists: Sequence[ScoredChunk],
    k: int = RRF_K,
) -> tuple[ScoredChunk, ...]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    RRF score for a document = sum(1 / (k + rank_i)) across all lists
    where rank_i is the 1-based rank in list i.
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    best_chunk: dict[str, ScoredChunk] = {}

    for results in result_lists:
        for rank, sc in enumerate(results, start=1):
            chunk_id = sc.chunk.chunk_id
            rrf_scores[chunk_id] += 1.0 / (k + rank)

            if chunk_id not in best_chunk:
                best_chunk[chunk_id] = sc

    sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)

    return tuple(
        ScoredChunk(
            chunk=best_chunk[cid].chunk,
            score=rrf_scores[cid],
            retrieval_method="hybrid",
        )
        for cid in sorted_ids
    )
