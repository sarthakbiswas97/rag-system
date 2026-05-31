from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag.api.middleware import RequestContextMiddleware
from rag.api.routes_metrics import router as metrics_router
from rag.observability.logging import setup_logging
from rag.observability.metrics import (
    ABSTENTION_COUNT,
    INGESTION_CHUNKS,
    QUERY_COUNT,
)


@pytest.fixture()
def metrics_client() -> TestClient:
    app = FastAPI()
    app.include_router(metrics_router)
    return TestClient(app)


class TestMetricsEndpoint:
    def test_returns_200(self, metrics_client: TestClient) -> None:
        resp = metrics_client.get("/metrics")
        assert resp.status_code == 200

    def test_content_type_is_text(self, metrics_client: TestClient) -> None:
        resp = metrics_client.get("/metrics")
        assert "text/plain" in resp.headers["content-type"]

    def test_contains_prometheus_format(self, metrics_client: TestClient) -> None:
        resp = metrics_client.get("/metrics")
        body = resp.text
        assert "rag_requests_total" in body or "# HELP" in body


class TestMetricsCounters:
    def test_query_count_increments(self) -> None:
        before = QUERY_COUNT.labels(tenant_id="t1")._value.get()
        QUERY_COUNT.labels(tenant_id="t1").inc()
        after = QUERY_COUNT.labels(tenant_id="t1")._value.get()
        assert after == before + 1

    def test_ingestion_chunks_increments(self) -> None:
        before = INGESTION_CHUNKS.labels(tenant_id="t1")._value.get()
        INGESTION_CHUNKS.labels(tenant_id="t1").inc(10)
        after = INGESTION_CHUNKS.labels(tenant_id="t1")._value.get()
        assert after == before + 10

    def test_abstention_count_increments(self) -> None:
        before = ABSTENTION_COUNT._value.get()
        ABSTENTION_COUNT.inc()
        after = ABSTENTION_COUNT._value.get()
        assert after == before + 1


class TestRequestMiddlewareMetrics:
    def test_middleware_records_request(self) -> None:
        app = FastAPI()
        app.add_middleware(RequestContextMiddleware)

        @app.get("/test")
        async def test_route() -> dict:
            return {"ok": True}

        client = TestClient(app)
        resp = client.get("/test")

        assert resp.status_code == 200
        assert "X-Request-ID" in resp.headers


class TestStructuredLogging:
    def test_setup_logging_runs(self) -> None:
        setup_logging("INFO")

    def test_setup_logging_with_debug(self) -> None:
        setup_logging("DEBUG")
