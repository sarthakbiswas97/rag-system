from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Sequence

import torch
from sentence_transformers import SentenceTransformer

from rag.models.document import Chunk

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(
        self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 64
    ) -> None:
        self._batch_size = batch_size
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = SentenceTransformer(model_name, device=device)
        self._dimension = self._model.get_embedding_dimension()

        logger.info(
            "Embedder initialized",
            extra={
                "model": model_name,
                "device": device,
                "dimension": self._dimension,
            },
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()

        start = time.perf_counter()
        embeddings = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "Embedded texts",
            extra={
                "count": len(texts),
                "elapsed_ms": round(elapsed_ms, 1),
            },
        )

        return tuple(tuple(float(x) for x in vec) for vec in embeddings)

    def embed_chunks(self, chunks: Sequence[Chunk]) -> tuple[Chunk, ...]:
        if not chunks:
            return ()

        texts = [c.text for c in chunks]
        embeddings = self.embed_texts(texts)

        return tuple(
            dataclasses.replace(chunk, embedding=emb)
            for chunk, emb in zip(chunks, embeddings, strict=True)
        )
