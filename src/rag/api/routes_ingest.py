from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from rag.api.dependencies import (
    get_ingest_rate_limiter,
    get_job_store,
    get_pipeline,
    get_query_cache,
)
from rag.api.rate_limiter import RateLimiter
from rag.api.schemas import IngestResponse, JobResponse
from rag.ingestion.job import IngestionJob, JobStatus, JobStore
from rag.ingestion.pipeline import IngestionPipeline
from rag.retrieval.cache import QueryCache
from rag.tenancy.auth import get_current_tenant
from rag.tenancy.models import Tenant

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/v1/ingest", response_model=IngestResponse)
async def ingest(
    files: list[UploadFile],
    tenant: Tenant = Depends(get_current_tenant),
    pipeline: IngestionPipeline = Depends(get_pipeline),
    query_cache: QueryCache | None = Depends(get_query_cache),
    rate_limiter: RateLimiter | None = Depends(get_ingest_rate_limiter),
) -> IngestResponse:
    if rate_limiter is not None:
        result = rate_limiter.check(tenant.id, "ingest")
        if not result.allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(result.reset_after)},
            )
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        paths: list[Path] = []
        for upload in files:
            dest = tmp_dir / (upload.filename or "unnamed")
            content = await upload.read()
            dest.write_bytes(content)
            paths.append(dest)

        result = pipeline.ingest_documents(paths, tenant_id=tenant.id)

        if query_cache is not None and result.chunks_created > 0:
            query_cache.invalidate_tenant(tenant.id)

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
    pipeline: IngestionPipeline = Depends(get_pipeline),
    job_store: JobStore = Depends(get_job_store),
) -> JobResponse:
    tmp_dir = Path(tempfile.mkdtemp())

    paths: list[Path] = []
    for upload in files:
        dest = tmp_dir / (upload.filename or "unnamed")
        content = await upload.read()
        dest.write_bytes(content)
        paths.append(dest)

    job = job_store.create(tenant_id=tenant.id)

    asyncio.get_event_loop().run_in_executor(
        None,
        _run_ingestion,
        pipeline,
        job_store,
        job,
        paths,
        tmp_dir,
        tenant.id,
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


def _run_ingestion(
    pipeline: IngestionPipeline,
    job_store: JobStore,
    job: IngestionJob,
    paths: list[Path],
    tmp_dir: Path,
    tenant_id: str,
) -> None:
    running_job = IngestionJob(
        job_id=job.job_id,
        tenant_id=job.tenant_id,
        status=JobStatus.RUNNING,
        created_at=job.created_at,
    )
    job_store.update(running_job)

    def _on_progress(
        docs_done: int, docs_total: int, chunks: int
    ) -> None:
        progress_job = IngestionJob(
            job_id=job.job_id,
            tenant_id=job.tenant_id,
            status=JobStatus.RUNNING,
            documents_processed=docs_done,
            chunks_created=chunks,
            created_at=job.created_at,
        )
        job_store.update(progress_job)

    try:
        result = pipeline.ingest_documents_batch(
            paths, tenant_id=tenant_id, on_progress=_on_progress
        )

        completed_job = IngestionJob(
            job_id=job.job_id,
            tenant_id=job.tenant_id,
            status=JobStatus.COMPLETED,
            documents_processed=result.documents_processed,
            documents_skipped=result.documents_skipped,
            documents_failed=result.documents_failed,
            chunks_created=result.chunks_created,
            elapsed_ms=result.elapsed_ms,
            created_at=job.created_at,
            completed_at=datetime.now(tz=UTC).isoformat(),
        )
        job_store.update(completed_job)

        logger.info(
            "Async ingestion completed",
            extra={"job_id": job.job_id, "chunks": result.chunks_created},
        )

    except Exception:
        logger.exception(
            "Async ingestion failed", extra={"job_id": job.job_id}
        )
        failed_job = IngestionJob(
            job_id=job.job_id,
            tenant_id=job.tenant_id,
            status=JobStatus.FAILED,
            error="Ingestion failed unexpectedly",
            created_at=job.created_at,
            completed_at=datetime.now(tz=UTC).isoformat(),
        )
        job_store.update(failed_job)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


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
