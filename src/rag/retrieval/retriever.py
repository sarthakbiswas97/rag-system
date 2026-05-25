from __future__ import annotations

import logging
import time

from rag.ingestion.embedder import Embedder
from rag.models.retrieval import RetrievalResult
from rag.retrieval.bm25_store import BM25Store
from rag.retrieval.hybrid import reciprocal_rank_fusion
from rag.retrieval.reranker import Reranker
from rag.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        reranker: Reranker | None = None,
        bm25_store: BM25Store | None = None,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._reranker = reranker
        self._bm25_store = bm25_store

    def retrieve(
        self, query: str, top_k: int = 5, tenant_id: str = ""
    ) -> RetrievalResult:
        start = time.perf_counter()

        query_embedding = self._embedder.embed_texts([query])[0]

        # Retrieve broader set when reranking, narrow set otherwise
        search_k = max(top_k * 10, 50) if self._reranker else top_k
        vector_results = self._vector_store.search(
            query_embedding, top_k=search_k, tenant_id=tenant_id
        )

        if self._bm25_store is not None:
            bm25_results = self._bm25_store.search(
                query, top_k=search_k, tenant_id=tenant_id
            )
            scored_chunks = reciprocal_rank_fusion(vector_results, bm25_results)
            if not self._reranker:
                scored_chunks = scored_chunks[:top_k]
        else:
            scored_chunks = vector_results

        if self._reranker and scored_chunks:
            scored_chunks = self._reranker.rerank(query, scored_chunks, top_k=top_k)

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "Retrieval complete",
            extra={
                "query": query[:100],
                "top_k": top_k,
                "hybrid": self._bm25_store is not None,
                "reranked": self._reranker is not None,
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
