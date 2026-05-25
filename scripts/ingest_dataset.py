#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from qdrant_client import QdrantClient

from rag.config import get_settings
from rag.ingestion.embedder import Embedder
from rag.ingestion.pipeline import IngestionPipeline
from rag.retrieval.vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest documents into the RAG vector store"
    )
    parser.add_argument(
        "--directory",
        type=Path,
        required=True,
        help="Directory containing documents to ingest",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Qdrant collection name (default: from config)",
    )
    args = parser.parse_args()

    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    collection = args.collection or settings.qdrant_collection

    embedder = Embedder(
        model_name=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )

    client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )
    store = VectorStore(client=client, collection=collection)
    store.create_collection(vector_size=embedder.dimension)

    pipeline = IngestionPipeline(
        embedder=embedder,
        vector_store=store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    result = pipeline.ingest_directory(args.directory)

    print(f"\nIngestion complete:")
    print(f"  Processed: {result.documents_processed}")
    print(f"  Skipped:   {result.documents_skipped}")
    print(f"  Failed:    {result.documents_failed}")
    print(f"  Chunks:    {result.chunks_created}")
    print(f"  Time:      {result.elapsed_ms:.0f}ms")


if __name__ == "__main__":
    main()
