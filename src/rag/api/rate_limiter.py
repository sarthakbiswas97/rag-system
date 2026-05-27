from __future__ import annotations

import logging
import time

from redis import Redis

logger = logging.getLogger(__name__)

RATE_LIMIT_PREFIX = "rl:"


class RateLimiter:
    """Redis-backed sliding window rate limiter.

    Uses sorted sets with timestamps for precise sliding window counting.
    Each tenant gets independent rate limits.
    """

    def __init__(
        self,
        client: Redis,
        max_requests: int = 60,
        window_seconds: int = 60,
    ) -> None:
        self._client = client
        self._max_requests = max_requests
        self._window_seconds = window_seconds

    def check(self, tenant_id: str, endpoint: str = "default") -> RateLimitResult:
        """Check if request is allowed and record it atomically."""
        key = f"{RATE_LIMIT_PREFIX}{tenant_id}:{endpoint}"
        now = time.time()
        window_start = now - self._window_seconds

        pipe = self._client.pipeline()
        # Remove expired entries
        pipe.zremrangebyscore(key, 0, window_start)
        # Count remaining entries in window
        pipe.zcard(key)
        # Add current request
        pipe.zadd(key, {f"{now}": now})
        # Set TTL to auto-clean
        pipe.expire(key, self._window_seconds)
        results = pipe.execute()

        current_count = results[1]
        remaining = max(0, self._max_requests - current_count - 1)
        allowed = current_count < self._max_requests

        if not allowed:
            # Remove the entry we just added since request is denied
            self._client.zrem(key, f"{now}")
            logger.warning(
                "Rate limit exceeded",
                extra={
                    "tenant_id": tenant_id,
                    "endpoint": endpoint,
                    "count": current_count,
                    "limit": self._max_requests,
                },
            )

        return RateLimitResult(
            allowed=allowed,
            limit=self._max_requests,
            remaining=remaining if allowed else 0,
            reset_after=self._window_seconds,
        )


class RateLimitResult:
    __slots__ = ("allowed", "limit", "remaining", "reset_after")

    def __init__(
        self,
        allowed: bool,
        limit: int,
        remaining: int,
        reset_after: int,
    ) -> None:
        self.allowed = allowed
        self.limit = limit
        self.remaining = remaining
        self.reset_after = reset_after
