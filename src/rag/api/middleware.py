from __future__ import annotations

import time
from uuid import uuid4

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from rag.observability.metrics import REQUEST_COUNT, REQUEST_LATENCY

logger = structlog.get_logger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = str(uuid4())
        start = time.perf_counter()

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        response = await call_next(request)

        elapsed = time.perf_counter() - start
        response.headers["X-Request-ID"] = request_id

        endpoint = request.url.path
        method = request.method
        status = str(response.status_code)

        REQUEST_COUNT.labels(
            method=method, endpoint=endpoint, status=status
        ).inc()
        REQUEST_LATENCY.labels(
            method=method, endpoint=endpoint
        ).observe(elapsed)

        logger.info(
            "request_completed",
            method=method,
            path=endpoint,
            status=response.status_code,
            elapsed_ms=round(elapsed * 1000, 1),
        )

        return response
