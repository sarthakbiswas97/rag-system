from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from rag.models.document import Chunk
from rag.models.retrieval import ScoredChunk

logger = logging.getLogger(__name__)

_TOKENIZE_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKENIZE_RE.findall(text.lower())


class BM25Store:
    """In-memory BM25 index with tenant isolation."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._corpus: list[list[str]] = []
        self._tenant_indices: dict[str, list[int]] = defaultdict(list)
        self._index: BM25Okapi | None = None

    @property
    def size(self) -> int:
        return len(self._chunks)

    def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        for chunk in chunks:
            idx = len(self._chunks)
            self._chunks.append(chunk)
            tokens = _tokenize(chunk.text)
            self._corpus.append(tokens)

            tenant_id = chunk.metadata.tenant_id
            if tenant_id:
                self._tenant_indices[tenant_id].append(idx)

        self._rebuild_index()

        logger.info(
            "BM25 index updated",
            extra={"total_chunks": len(self._chunks), "added": len(chunks)},
        )

    def _rebuild_index(self) -> None:
        if self._corpus:
            self._index = BM25Okapi(self._corpus)
        else:
            self._index = None

    def search(
        self,
        query: str,
        top_k: int = 50,
        tenant_id: str = "",
    ) -> tuple[ScoredChunk, ...]:
        if self._index is None or not self._chunks:
            return ()

        tokens = _tokenize(query)
        if not tokens:
            return ()

        scores = self._index.get_scores(tokens)

        if tenant_id:
            allowed = set(self._tenant_indices.get(tenant_id, []))
            if not allowed:
                return ()
            candidates = [
                (idx, float(scores[idx]))
                for idx in allowed
                if scores[idx] != 0.0
            ]
        else:
            candidates = [
                (idx, float(score))
                for idx, score in enumerate(scores)
                if score != 0.0
            ]

        candidates.sort(key=lambda x: x[1], reverse=True)
        top = candidates[:top_k]

        return tuple(
            ScoredChunk(
                chunk=self._chunks[idx],
                score=score,
                retrieval_method="bm25",
            )
            for idx, score in top
        )

    def clear(self) -> None:
        self._chunks.clear()
        self._corpus.clear()
        self._tenant_indices.clear()
        self._index = None
