from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from redis import Redis

logger = logging.getLogger(__name__)

JOB_TTL_SECONDS = 86400  # 24 hours
JOB_PREFIX = "ingest_job:"


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class IngestionJob:
    job_id: str = field(default_factory=lambda: str(uuid4()))
    tenant_id: str = ""
    status: JobStatus = JobStatus.PENDING
    documents_processed: int = 0
    documents_skipped: int = 0
    documents_failed: int = 0
    chunks_created: int = 0
    elapsed_ms: float = 0.0
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    completed_at: str | None = None


def _job_key(job_id: str) -> str:
    return f"{JOB_PREFIX}{job_id}"


def _job_to_dict(job: IngestionJob) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "tenant_id": job.tenant_id,
        "status": job.status.value,
        "documents_processed": job.documents_processed,
        "documents_skipped": job.documents_skipped,
        "documents_failed": job.documents_failed,
        "chunks_created": job.chunks_created,
        "elapsed_ms": job.elapsed_ms,
        "error": job.error,
        "created_at": job.created_at,
        "completed_at": job.completed_at,
    }


def _dict_to_job(data: dict[str, Any]) -> IngestionJob:
    return IngestionJob(
        job_id=data["job_id"],
        tenant_id=data.get("tenant_id", ""),
        status=JobStatus(data["status"]),
        documents_processed=data.get("documents_processed", 0),
        documents_skipped=data.get("documents_skipped", 0),
        documents_failed=data.get("documents_failed", 0),
        chunks_created=data.get("chunks_created", 0),
        elapsed_ms=data.get("elapsed_ms", 0.0),
        error=data.get("error"),
        created_at=data.get("created_at", ""),
        completed_at=data.get("completed_at"),
    )


class JobStore:
    """Redis-backed ingestion job tracker."""

    def __init__(self, client: Redis, ttl_seconds: int = JOB_TTL_SECONDS) -> None:
        self._client = client
        self._ttl = ttl_seconds

    def create(self, tenant_id: str = "") -> IngestionJob:
        job = IngestionJob(tenant_id=tenant_id)
        self._save(job)
        logger.info(
            "Ingestion job created",
            extra={"job_id": job.job_id, "tenant_id": tenant_id},
        )
        return job

    def get(self, job_id: str) -> IngestionJob | None:
        key = _job_key(job_id)
        raw = self._client.get(key)
        if raw is None:
            return None
        return _dict_to_job(json.loads(raw))

    def update(self, job: IngestionJob) -> None:
        self._save(job)

    def _save(self, job: IngestionJob) -> None:
        key = _job_key(job.job_id)
        data = json.dumps(_job_to_dict(job))
        self._client.setex(key, self._ttl, data)
