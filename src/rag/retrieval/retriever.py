from __future__ import annotations

import logging
import time

from rag.ingestion.embedder import Embedder
from rag.models.retrieval import RetrievalResult
from rag.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 50) -> RetrievalResult:
        start = time.perf_counter()

        query_embedding = self._embedder.embed_texts([query])[0]
        scored_chunks = self._vector_store.search(query_embedding, top_k=top_k)

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "Retrieval complete",
            extra={
                "query": query[:100],
                "top_k": top_k,
                "results": len(scored_chunks),
                "elapsed_ms": round(elapsed_ms, 1),
            },
        )

        return RetrievalResult(
            query=query,
            rewritten_query=None,
            scored_chunks=scored_chunks,
            retrieval_time_ms=round(elapsed_ms, 1),
        )
