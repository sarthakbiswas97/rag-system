from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from rag.api.dependencies import (
    get_db_session,
    get_ingest_rate_limiter,
    get_ingestion_worker,
    get_job_store,
    get_pipeline,
    get_query_cache,
)
from rag.api.rate_limiter import RateLimiter
from rag.api.schemas import IngestResponse, JobResponse
from rag.ingestion.job import IngestionJob, JobStore
from rag.ingestion.pipeline import IngestionPipeline
from rag.ingestion.worker import IngestionWorker
from rag.retrieval.cache import QueryCache
from rag.tenancy.auth import get_current_tenant
from rag.tenancy.models import EventType, Tenant
from rag.tenancy.usage import UsageTracker

logger = logging.getLogger(__name__)

router = APIRouter()


async def _save_uploads(files: list[UploadFile]) -> tuple[list[Path], Path]:
    tmp_dir = Path(tempfile.mkdtemp())
    paths: list[Path] = []
    for upload in files:
        dest = tmp_dir / (upload.filename or "unnamed")
        content = await upload.read()
        dest.write_bytes(content)
        paths.append(dest)
    return paths, tmp_dir


@router.post("/v1/ingest", response_model=IngestResponse)
async def ingest(
    files: list[UploadFile],
    tenant: Tenant = Depends(get_current_tenant),
    pipeline: IngestionPipeline = Depends(get_pipeline),
    query_cache: QueryCache | None = Depends(get_query_cache),
    rate_limiter: RateLimiter | None = Depends(get_ingest_rate_limiter),
    session: AsyncSession = Depends(get_db_session),
) -> IngestResponse:
    if rate_limiter is not None:
        result = rate_limiter.check(tenant.id, "ingest")
        if not result.allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(result.reset_after)},
            )

    paths, tmp_dir = await _save_uploads(files)
    try:
        result = pipeline.ingest_documents(paths, tenant_id=tenant.id)

        if query_cache is not None and result.chunks_created > 0:
            query_cache.invalidate_tenant(tenant.id)

        tracker = UsageTracker(session)
        await tracker.record(
            tenant_id=tenant.id,
            event_type=EventType.INGEST,
            value=result.chunks_created,
            elapsed_ms=result.elapsed_ms,
        )

        return IngestResponse(
            documents_processed=result.documents_processed,
            documents_skipped=result.documents_skipped,
            documents_failed=result.documents_failed,
            chunks_created=result.chunks_created,
            elapsed_ms=result.elapsed_ms,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.post("/v1/ingest/async", response_model=JobResponse)
async def ingest_async(
    files: list[UploadFile],
    tenant: Tenant = Depends(get_current_tenant),
    job_store: JobStore = Depends(get_job_store),
    worker: IngestionWorker = Depends(get_ingestion_worker),
    rate_limiter: RateLimiter | None = Depends(get_ingest_rate_limiter),
) -> JobResponse:
    if rate_limiter is not None:
        result = rate_limiter.check(tenant.id, "ingest")
        if not result.allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(result.reset_after)},
            )

    paths, tmp_dir = await _save_uploads(files)
    job = job_store.create(tenant_id=tenant.id)

    worker.submit(
        job=job,
        paths=paths,
        tmp_dir=tmp_dir,
        tenant_id=tenant.id,
    )

    return _job_to_response(job)


@router.get("/v1/ingest/jobs/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    tenant: Tenant = Depends(get_current_tenant),
    job_store: JobStore = Depends(get_job_store),
) -> JobResponse:
    job = job_store.get(job_id)
    if job is None or job.tenant_id != tenant.id:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_to_response(job)


def _job_to_response(job: IngestionJob) -> JobResponse:
    return JobResponse(
        job_id=job.job_id,
        status=job.status.value,
        documents_processed=job.documents_processed,
        documents_skipped=job.documents_skipped,
        documents_failed=job.documents_failed,
        chunks_created=job.chunks_created,
        elapsed_ms=job.elapsed_ms,
        error=job.error,
        created_at=job.created_at,
        completed_at=job.completed_at,
    )
