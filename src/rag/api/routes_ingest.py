from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile

from rag.api.dependencies import get_pipeline
from rag.api.schemas import IngestResponse
from rag.ingestion.pipeline import IngestionPipeline

router = APIRouter()


@router.post("/v1/ingest", response_model=IngestResponse)
async def ingest(
    files: list[UploadFile],
    pipeline: IngestionPipeline = Depends(get_pipeline),
) -> IngestResponse:
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        paths: list[Path] = []
        for upload in files:
            dest = tmp_dir / (upload.filename or "unnamed")
            content = await upload.read()
            dest.write_bytes(content)
            paths.append(dest)

        result = pipeline.ingest_documents(paths)

        return IngestResponse(
            documents_processed=result.documents_processed,
            documents_skipped=result.documents_skipped,
            documents_failed=result.documents_failed,
            chunks_created=result.chunks_created,
            elapsed_ms=result.elapsed_ms,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
