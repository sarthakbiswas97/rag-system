from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass

from qdrant_client.models import SparseVector

logger = logging.getLogger(__name__)

_TOKENIZE_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKENIZE_RE.findall(text.lower())


@dataclass(frozen=True)
class SparseEmbedding:
    indices: tuple[int, ...]
    values: tuple[float, ...]

    def to_qdrant(self) -> SparseVector:
        return SparseVector(indices=list(self.indices), values=list(self.values))


class SparseEmbedder:
    """Builds term-frequency sparse vectors with a bounded vocabulary.

    Each token is mapped to an integer index. Vocabulary grows incrementally
    up to ``max_vocab_size``, after which unseen tokens are ignored.
    Query vectors use sublinear TF scaling (1 + log(tf)) to dampen
    frequent terms.
    """

    def __init__(self, max_vocab_size: int = 50000) -> None:
        self._max_vocab_size = max_vocab_size
        self._vocab: dict[str, int] = {}
        self._next_idx = 0

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def _get_or_add_index(self, token: str) -> int | None:
        idx = self._vocab.get(token)
        if idx is not None:
            return idx
        if self._next_idx >= self._max_vocab_size:
            return None
        idx = self._next_idx
        self._vocab[token] = idx
        self._next_idx += 1
        return idx

    def embed_text(self, text: str) -> SparseEmbedding:
        tokens = _tokenize(text)
        if not tokens:
            return SparseEmbedding(indices=(), values=())

        counts = Counter(tokens)
        indices: list[int] = []
        values: list[float] = []

        for token, tf in counts.items():
            idx = self._vocab.get(token)
            if idx is None:
                idx = self._get_or_add_index(token)
            if idx is not None:
                indices.append(idx)
                values.append(float(tf))

        # Sort by index for deterministic output
        sorted_pairs = sorted(zip(indices, values, strict=False), key=lambda x: x[0])
        return SparseEmbedding(
            indices=tuple(i for i, _ in sorted_pairs),
            values=tuple(v for _, v in sorted_pairs),
        )

    def embed_texts(self, texts: list[str]) -> list[SparseEmbedding]:
        return [self.embed_text(t) for t in texts]

    def embed_query(self, text: str) -> SparseEmbedding:
        """Sublinear TF scaling for queries: value = 1 + log(tf)."""
        tokens = _tokenize(text)
        if not tokens:
            return SparseEmbedding(indices=(), values=())

        counts = Counter(tokens)
        indices: list[int] = []
        values: list[float] = []

        for token, tf in counts.items():
            idx = self._vocab.get(token)
            if idx is not None:
                indices.append(idx)
                values.append(1.0 + math.log(tf))

        sorted_pairs = sorted(zip(indices, values, strict=False), key=lambda x: x[0])
        return SparseEmbedding(
            indices=tuple(i for i, _ in sorted_pairs),
            values=tuple(v for _, v in sorted_pairs),
        )

    def clear(self) -> None:
        self._vocab.clear()
        self._next_idx = 0
