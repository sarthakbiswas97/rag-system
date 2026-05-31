from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from rag.ingestion.chunker import chunk_document
from rag.ingestion.embedder import Embedder
from rag.ingestion.hasher import ContentHasher, compute_hash
from rag.ingestion.loader import SUPPORTED_EXTENSIONS, load_document
from rag.models.document import Chunk
from rag.models.ingestion import DocumentInfo, IngestionResult
from rag.retrieval.sparse_embedder import SparseEmbedder
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
        sparse_embedder: SparseEmbedder | None = None,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._hasher = ContentHasher()
        self._sparse_embedder = sparse_embedder

    def ingest_documents(
        self, paths: Sequence[Path], tenant_id: str = ""
    ) -> IngestionResult:
        start = time.perf_counter()
        processed = 0
        skipped = 0
        failed = 0
        total_chunks = 0
        processed_docs: list[DocumentInfo] = []

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
                embedded = self._attach_sparse_embeddings(embedded)
                self._vector_store.upsert_chunks(embedded)
                self._upsert_sparse_vectors(embedded)

                chunk_count = len(embedded)
                processed += 1
                total_chunks += chunk_count
                processed_docs.append(
                    DocumentInfo(
                        document_id=doc.document_id,
                        source_file=Path(doc.source_path).name,
                        content_hash=compute_hash(doc.content),
                        chunk_count=chunk_count,
                    )
                )

                logger.info(
                    "Ingested document",
                    extra={
                        "path": str(path),
                        "chunks": chunk_count,
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
            processed_documents=tuple(processed_docs),
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
        processed_ids: list[str] = []
        doc_meta: dict[str, tuple[str, str]] = {}

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
                processed_ids.append(doc.document_id)
                doc_meta[doc.document_id] = (
                    Path(doc.source_path).name,
                    compute_hash(doc.content),
                )

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
        embedded = self._attach_sparse_embeddings(embedded)
        self._vector_store.upsert_chunks(embedded)
        self._upsert_sparse_vectors(embedded)

        total_chunks = len(embedded)

        if on_progress is not None:
            on_progress(total_docs, total_docs, total_chunks)

        # Compute per-document chunk counts from the (restored) embedded chunks
        from collections import Counter

        chunk_counts = Counter(c.document_id for c in embedded)
        processed_docs = tuple(
            DocumentInfo(
                document_id=doc_id,
                source_file=doc_meta[doc_id][0],
                content_hash=doc_meta[doc_id][1],
                chunk_count=chunk_counts[doc_id],
            )
            for doc_id in processed_ids
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        result = IngestionResult(
            documents_processed=processed,
            documents_skipped=skipped,
            documents_failed=failed,
            chunks_created=total_chunks,
            elapsed_ms=round(elapsed_ms, 1),
            processed_documents=processed_docs,
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

    def _attach_sparse_embeddings(self, chunks: tuple[Chunk, ...]) -> tuple[Chunk, ...]:
        if self._sparse_embedder is None:
            return chunks
        sparse_embeddings = self._sparse_embedder.embed_texts([c.text for c in chunks])
        return tuple(
            Chunk(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                text=c.text,
                metadata=c.metadata,
                embedding=c.embedding,
                sparse_embedding=se.to_qdrant(),
            )
            for c, se in zip(chunks, sparse_embeddings, strict=True)
        )

    def _upsert_sparse_vectors(self, chunks: tuple[Chunk, ...]) -> None:
        if self._sparse_embedder is None:
            return
        chunk_ids = [c.chunk_id for c in chunks if c.sparse_embedding is not None]
        sparse_vectors = [
            c.sparse_embedding for c in chunks if c.sparse_embedding is not None
        ]
        if chunk_ids:
            self._vector_store.upsert_sparse_vectors(chunk_ids, sparse_vectors)

    def ingest_directory(self, directory: Path, tenant_id: str = "") -> IngestionResult:
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
