from __future__ import annotations

import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from redis import Redis

from rag.ingestion.job import IngestionJob, JobStatus, JobStore
from rag.ingestion.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2.0
DLQ_PREFIX = "dlq:ingest:"
DLQ_TTL_SECONDS = 86400 * 7  # 7 days
TENANT_SEMAPHORE_PREFIX = "sem:ingest:"
MAX_CONCURRENT_PER_TENANT = 3


@dataclass(frozen=True)
class WorkerConfig:
    max_workers: int = 4
    max_retries: int = MAX_RETRIES
    max_concurrent_per_tenant: int = MAX_CONCURRENT_PER_TENANT


class IngestionWorker:
    """Thread-pool based async ingestion worker with retries and tenant isolation."""

    def __init__(
        self,
        pipeline: IngestionPipeline,
        job_store: JobStore,
        redis_client: Redis,
        config: WorkerConfig | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._job_store = job_store
        self._redis = redis_client
        self._config = config or WorkerConfig()
        self._executor = ThreadPoolExecutor(
            max_workers=self._config.max_workers,
            thread_name_prefix="ingest-worker",
        )

    def submit(
        self,
        job: IngestionJob,
        paths: list[Path],
        tmp_dir: Path,
        tenant_id: str,
    ) -> None:
        self._executor.submit(
            self._execute_with_retry,
            job,
            paths,
            tmp_dir,
            tenant_id,
        )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)

    def _acquire_tenant_slot(self, tenant_id: str) -> bool:
        key = f"{TENANT_SEMAPHORE_PREFIX}{tenant_id}"
        current = self._redis.incr(key)
        if current == 1:
            self._redis.expire(key, 3600)
        if current > self._config.max_concurrent_per_tenant:
            self._redis.decr(key)
            return False
        return True

    def _release_tenant_slot(self, tenant_id: str) -> None:
        key = f"{TENANT_SEMAPHORE_PREFIX}{tenant_id}"
        self._redis.decr(key)

    def _move_to_dlq(self, job: IngestionJob, error: str) -> None:
        import json

        dlq_key = f"{DLQ_PREFIX}{job.job_id}"
        data = json.dumps(
            {
                "job_id": job.job_id,
                "tenant_id": job.tenant_id,
                "error": error,
                "failed_at": datetime.now(tz=UTC).isoformat(),
            }
        )
        self._redis.setex(dlq_key, DLQ_TTL_SECONDS, data)
        logger.warning(
            "Job moved to dead-letter queue",
            extra={"job_id": job.job_id, "tenant_id": job.tenant_id},
        )

    def _execute_with_retry(
        self,
        job: IngestionJob,
        paths: list[Path],
        tmp_dir: Path,
        tenant_id: str,
    ) -> None:
        if not self._acquire_tenant_slot(tenant_id):
            logger.warning(
                "Tenant concurrency limit reached, queuing",
                extra={"tenant_id": tenant_id, "job_id": job.job_id},
            )
            # Wait briefly and retry acquiring
            time.sleep(RETRY_DELAY_SECONDS)
            if not self._acquire_tenant_slot(tenant_id):
                failed_job = IngestionJob(
                    job_id=job.job_id,
                    tenant_id=job.tenant_id,
                    status=JobStatus.FAILED,
                    error="Too many concurrent ingestion jobs",
                    created_at=job.created_at,
                    completed_at=datetime.now(tz=UTC).isoformat(),
                )
                self._job_store.update(failed_job)
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return

        try:
            self._run_with_retries(job, paths, tmp_dir, tenant_id)
        finally:
            self._release_tenant_slot(tenant_id)

    def _run_with_retries(
        self,
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
        self._job_store.update(running_job)

        last_error = ""
        for attempt in range(1, self._config.max_retries + 1):
            try:
                result = self._pipeline.ingest_documents(paths, tenant_id=tenant_id)

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
                self._job_store.update(completed_job)

                logger.info(
                    "Async ingestion completed",
                    extra={
                        "job_id": job.job_id,
                        "chunks": result.chunks_created,
                        "attempt": attempt,
                    },
                )
                return

            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "Ingestion attempt failed",
                    extra={
                        "job_id": job.job_id,
                        "attempt": attempt,
                        "max_retries": self._config.max_retries,
                        "error": last_error,
                    },
                )
                if attempt < self._config.max_retries:
                    delay = RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
                    time.sleep(delay)

        # All retries exhausted
        failed_job = IngestionJob(
            job_id=job.job_id,
            tenant_id=job.tenant_id,
            status=JobStatus.FAILED,
            error=f"Failed after {self._config.max_retries} attempts: {last_error}",
            created_at=job.created_at,
            completed_at=datetime.now(tz=UTC).isoformat(),
        )
        self._job_store.update(failed_job)
        self._move_to_dlq(job, last_error)

        logger.error(
            "Async ingestion failed permanently",
            extra={"job_id": job.job_id, "tenant_id": tenant_id},
        )

        shutil.rmtree(tmp_dir, ignore_errors=True)
