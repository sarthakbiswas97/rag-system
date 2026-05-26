from __future__ import annotations

from unittest.mock import MagicMock

from rag.ingestion.job import (
    IngestionJob,
    JobStatus,
    JobStore,
    _dict_to_job,
    _job_to_dict,
)


def _make_mock_redis() -> MagicMock:
    client = MagicMock()
    storage: dict[str, str] = {}

    def mock_setex(key: str, ttl: int, value: str) -> None:
        storage[key] = value

    def mock_get(key: str) -> str | None:
        return storage.get(key)

    client.setex = mock_setex
    client.get = mock_get
    return client


class TestJobStore:
    def test_create_job(self) -> None:
        store = JobStore(client=_make_mock_redis())
        job = store.create(tenant_id="t1")

        assert job.job_id
        assert job.tenant_id == "t1"
        assert job.status == JobStatus.PENDING

    def test_get_job(self) -> None:
        store = JobStore(client=_make_mock_redis())
        created = store.create(tenant_id="t1")

        retrieved = store.get(created.job_id)

        assert retrieved is not None
        assert retrieved.job_id == created.job_id
        assert retrieved.status == JobStatus.PENDING

    def test_get_nonexistent_returns_none(self) -> None:
        store = JobStore(client=_make_mock_redis())
        assert store.get("nonexistent") is None

    def test_update_job(self) -> None:
        store = JobStore(client=_make_mock_redis())
        job = store.create()

        completed = IngestionJob(
            job_id=job.job_id,
            status=JobStatus.COMPLETED,
            documents_processed=5,
            chunks_created=25,
            elapsed_ms=1500.0,
        )
        store.update(completed)

        retrieved = store.get(job.job_id)
        assert retrieved is not None
        assert retrieved.status == JobStatus.COMPLETED
        assert retrieved.documents_processed == 5
        assert retrieved.chunks_created == 25

    def test_update_to_failed(self) -> None:
        store = JobStore(client=_make_mock_redis())
        job = store.create()

        failed = IngestionJob(
            job_id=job.job_id,
            status=JobStatus.FAILED,
            error="Something went wrong",
        )
        store.update(failed)

        retrieved = store.get(job.job_id)
        assert retrieved is not None
        assert retrieved.status == JobStatus.FAILED
        assert retrieved.error == "Something went wrong"

    def test_status_transitions(self) -> None:
        store = JobStore(client=_make_mock_redis())
        job = store.create()
        assert store.get(job.job_id).status == JobStatus.PENDING

        running = IngestionJob(
            job_id=job.job_id, status=JobStatus.RUNNING
        )
        store.update(running)
        assert store.get(job.job_id).status == JobStatus.RUNNING

        completed = IngestionJob(
            job_id=job.job_id, status=JobStatus.COMPLETED
        )
        store.update(completed)
        assert store.get(job.job_id).status == JobStatus.COMPLETED


class TestJobSerialization:
    def test_roundtrip(self) -> None:
        job = IngestionJob(
            tenant_id="t1",
            status=JobStatus.COMPLETED,
            documents_processed=3,
            documents_skipped=1,
            documents_failed=0,
            chunks_created=15,
            elapsed_ms=500.0,
            completed_at="2026-01-01T00:00:00",
        )

        data = _job_to_dict(job)
        restored = _dict_to_job(data)

        assert restored.job_id == job.job_id
        assert restored.tenant_id == "t1"
        assert restored.status == JobStatus.COMPLETED
        assert restored.documents_processed == 3
        assert restored.chunks_created == 15
        assert restored.completed_at == "2026-01-01T00:00:00"

    def test_roundtrip_with_error(self) -> None:
        job = IngestionJob(
            status=JobStatus.FAILED,
            error="Pipeline crashed",
        )

        data = _job_to_dict(job)
        restored = _dict_to_job(data)

        assert restored.status == JobStatus.FAILED
        assert restored.error == "Pipeline crashed"

    def test_roundtrip_pending(self) -> None:
        job = IngestionJob()
        data = _job_to_dict(job)
        restored = _dict_to_job(data)

        assert restored.status == JobStatus.PENDING
        assert restored.error is None
        assert restored.completed_at is None
