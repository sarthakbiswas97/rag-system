from __future__ import annotations

import logging
import time
from collections.abc import Sequence

import torch
from sentence_transformers import CrossEncoder

from rag.models.retrieval import ScoredChunk

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = CrossEncoder(model_name, device=device)

        logger.info(
            "Reranker initialized",
            extra={"model": model_name, "device": device},
        )

    def rerank(
        self,
        query: str,
        scored_chunks: Sequence[ScoredChunk],
        top_k: int = 5,
    ) -> tuple[ScoredChunk, ...]:
        if not scored_chunks:
            return ()

        pairs = [(query, sc.chunk.text) for sc in scored_chunks]

        start = time.perf_counter()
        scores = self._model.predict(pairs)
        elapsed_ms = (time.perf_counter() - start) * 1000

        reranked = sorted(
            zip(scored_chunks, scores, strict=True),
            key=lambda x: float(x[1]),
            reverse=True,
        )

        result = tuple(
            ScoredChunk(
                chunk=sc.chunk,
                score=float(score),
                retrieval_method="reranked",
            )
            for sc, score in reranked[:top_k]
        )

        logger.info(
            "Reranking complete",
            extra={
                "input_chunks": len(scored_chunks),
                "output_chunks": len(result),
                "elapsed_ms": round(elapsed_ms, 1),
            },
        )

        return result
