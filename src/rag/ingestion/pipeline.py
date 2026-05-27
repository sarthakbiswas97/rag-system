from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from rag.ingestion.chunker import chunk_document
from rag.ingestion.embedder import Embedder
from rag.ingestion.hasher import ContentHasher
from rag.ingestion.loader import SUPPORTED_EXTENSIONS, load_document
from rag.models.document import Chunk
from rag.models.ingestion import IngestionResult
from rag.retrieval.bm25_store import BM25Store
from rag.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Callback type: (documents_processed, documents_total, chunks_so_far)
ProgressCallback = Callable[[int, int, int], None]


class IngestionPipeline:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        bm25_store: BM25Store | None = None,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._hasher = ContentHasher()
        self._bm25_store = bm25_store

    def ingest_documents(
        self, paths: Sequence[Path], tenant_id: str = ""
    ) -> IngestionResult:
        start = time.perf_counter()
        processed = 0
        skipped = 0
        failed = 0
        total_chunks = 0

        for path in paths:
            try:
                doc = load_document(path)

                if self._hasher.is_duplicate(doc, tenant_id=tenant_id):
                    logger.info(
                        "Skipping duplicate document",
                        extra={"path": str(path)},
                    )
                    skipped += 1
                    continue

                chunks = chunk_document(
                    doc, self._chunk_size, self._chunk_overlap, tenant_id=tenant_id
                )
                if not chunks:
                    logger.info(
                        "Document produced no chunks",
                        extra={"path": str(path)},
                    )
                    skipped += 1
                    continue

                embedded = self._embedder.embed_chunks(chunks)
                self._vector_store.upsert_chunks(embedded)

                if self._bm25_store is not None:
                    self._bm25_store.add_chunks(embedded)

                processed += 1
                total_chunks += len(embedded)

                logger.info(
                    "Ingested document",
                    extra={
                        "path": str(path),
                        "chunks": len(embedded),
                    },
                )

            except Exception:
                logger.exception(
                    "Failed to ingest document",
                    extra={"path": str(path)},
                )
                failed += 1

        elapsed_ms = (time.perf_counter() - start) * 1000

        result = IngestionResult(
            documents_processed=processed,
            documents_skipped=skipped,
            documents_failed=failed,
            chunks_created=total_chunks,
            elapsed_ms=round(elapsed_ms, 1),
        )

        logger.info(
            "Ingestion complete",
            extra={
                "processed": result.documents_processed,
                "skipped": result.documents_skipped,
                "failed": result.documents_failed,
                "chunks": result.chunks_created,
                "elapsed_ms": result.elapsed_ms,
            },
        )

        return result

    def ingest_documents_batch(
        self,
        paths: Sequence[Path],
        tenant_id: str = "",
        on_progress: ProgressCallback | None = None,
    ) -> IngestionResult:
        """Batch-optimized ingestion: collect all chunks, embed together.

        Collects chunks from all documents first, sorts by text length
        to minimize padding waste during embedding, then embeds and
        upserts in a single pass. Significantly faster for multi-doc
        ingestion compared to per-document processing.
        """
        start = time.perf_counter()
        processed = 0
        skipped = 0
        failed = 0
        total_docs = len(paths)

        # Phase 1: Load and chunk all documents
        all_chunks: list[Chunk] = []
        for path in paths:
            try:
                doc = load_document(path)

                if self._hasher.is_duplicate(doc, tenant_id=tenant_id):
                    logger.info(
                        "Skipping duplicate document",
                        extra={"path": str(path)},
                    )
                    skipped += 1
                    continue

                chunks = chunk_document(
                    doc,
                    self._chunk_size,
                    self._chunk_overlap,
                    tenant_id=tenant_id,
                )
                if not chunks:
                    skipped += 1
                    continue

                all_chunks.extend(chunks)
                processed += 1

                if on_progress is not None:
                    on_progress(processed + skipped + failed, total_docs, 0)

            except Exception:
                logger.exception(
                    "Failed to load/chunk document",
                    extra={"path": str(path)},
                )
                failed += 1

        if not all_chunks:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return IngestionResult(
                documents_processed=processed,
                documents_skipped=skipped,
                documents_failed=failed,
                chunks_created=0,
                elapsed_ms=round(elapsed_ms, 1),
            )

        # Phase 2: Sort chunks by text length to minimize padding
        sort_indices = sorted(
            range(len(all_chunks)),
            key=lambda i: len(all_chunks[i].text),
        )
        sorted_chunks = tuple(all_chunks[i] for i in sort_indices)

        # Phase 3: Embed all chunks in optimized batches
        logger.info(
            "Embedding chunks in batch",
            extra={"chunk_count": len(sorted_chunks)},
        )
        embedded_sorted = self._embedder.embed_chunks(sorted_chunks)

        # Phase 4: Restore original order for consistency
        reverse_map = [0] * len(sort_indices)
        for new_idx, original_idx in enumerate(sort_indices):
            reverse_map[original_idx] = new_idx
        embedded = tuple(
            embedded_sorted[reverse_map[i]] for i in range(len(all_chunks))
        )

        # Phase 5: Upsert to vector store
        self._vector_store.upsert_chunks(embedded)

        if self._bm25_store is not None:
            self._bm25_store.add_chunks(embedded)

        total_chunks = len(embedded)

        if on_progress is not None:
            on_progress(total_docs, total_docs, total_chunks)

        elapsed_ms = (time.perf_counter() - start) * 1000

        result = IngestionResult(
            documents_processed=processed,
            documents_skipped=skipped,
            documents_failed=failed,
            chunks_created=total_chunks,
            elapsed_ms=round(elapsed_ms, 1),
        )

        logger.info(
            "Batch ingestion complete",
            extra={
                "processed": result.documents_processed,
                "skipped": result.documents_skipped,
                "failed": result.documents_failed,
                "chunks": result.chunks_created,
                "elapsed_ms": result.elapsed_ms,
            },
        )

        return result

    def ingest_directory(
        self, directory: Path, tenant_id: str = ""
    ) -> IngestionResult:
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        paths = sorted(
            p
            for p in directory.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        logger.info(
            "Found documents to ingest",
            extra={"directory": str(directory), "count": len(paths)},
        )

        return self.ingest_documents(paths, tenant_id=tenant_id)
