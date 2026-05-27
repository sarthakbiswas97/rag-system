from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rag.api.rate_limiter import RATE_LIMIT_PREFIX, RateLimiter


@pytest.fixture()
def mock_redis() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def limiter(mock_redis: MagicMock) -> RateLimiter:
    return RateLimiter(client=mock_redis, max_requests=5, window_seconds=60)


class TestRateLimiter:
    def test_allows_under_limit(
        self, limiter: RateLimiter, mock_redis: MagicMock
    ) -> None:
        pipe = MagicMock()
        pipe.execute.return_value = [None, 2, None, None]
        mock_redis.pipeline.return_value = pipe

        result = limiter.check("t1", "query")
        assert result.allowed is True
        assert result.remaining == 2
        assert result.limit == 5

    def test_blocks_at_limit(
        self, limiter: RateLimiter, mock_redis: MagicMock
    ) -> None:
        pipe = MagicMock()
        pipe.execute.return_value = [None, 5, None, None]
        mock_redis.pipeline.return_value = pipe

        result = limiter.check("t1", "query")
        assert result.allowed is False
        assert result.remaining == 0

    def test_blocks_over_limit(
        self, limiter: RateLimiter, mock_redis: MagicMock
    ) -> None:
        pipe = MagicMock()
        pipe.execute.return_value = [None, 10, None, None]
        mock_redis.pipeline.return_value = pipe

        result = limiter.check("t1", "query")
        assert result.allowed is False

    def test_removes_entry_on_deny(
        self, limiter: RateLimiter, mock_redis: MagicMock
    ) -> None:
        pipe = MagicMock()
        pipe.execute.return_value = [None, 5, None, None]
        mock_redis.pipeline.return_value = pipe

        limiter.check("t1", "query")
        mock_redis.zrem.assert_called_once()

    def test_does_not_remove_entry_on_allow(
        self, limiter: RateLimiter, mock_redis: MagicMock
    ) -> None:
        pipe = MagicMock()
        pipe.execute.return_value = [None, 0, None, None]
        mock_redis.pipeline.return_value = pipe

        limiter.check("t1", "query")
        mock_redis.zrem.assert_not_called()

    def test_reset_after_matches_window(
        self, limiter: RateLimiter, mock_redis: MagicMock
    ) -> None:
        pipe = MagicMock()
        pipe.execute.return_value = [None, 0, None, None]
        mock_redis.pipeline.return_value = pipe

        result = limiter.check("t1", "query")
        assert result.reset_after == 60

    def test_different_endpoints_use_different_keys(
        self, limiter: RateLimiter, mock_redis: MagicMock
    ) -> None:
        pipe = MagicMock()
        pipe.execute.return_value = [None, 0, None, None]
        mock_redis.pipeline.return_value = pipe

        limiter.check("t1", "query")
        limiter.check("t1", "ingest")

        calls = pipe.zremrangebyscore.call_args_list
        keys = [c[0][0] for c in calls]
        assert keys[0] != keys[1]
        assert "query" in keys[0]
        assert "ingest" in keys[1]

    def test_key_includes_prefix(
        self, limiter: RateLimiter, mock_redis: MagicMock
    ) -> None:
        pipe = MagicMock()
        pipe.execute.return_value = [None, 0, None, None]
        mock_redis.pipeline.return_value = pipe

        limiter.check("t1", "query")

        key = pipe.zremrangebyscore.call_args[0][0]
        assert key.startswith(RATE_LIMIT_PREFIX)


class TestRateLimiterCustomConfig:
    def test_custom_max_requests(self, mock_redis: MagicMock) -> None:
        limiter = RateLimiter(
            client=mock_redis, max_requests=100, window_seconds=30
        )
        pipe = MagicMock()
        pipe.execute.return_value = [None, 50, None, None]
        mock_redis.pipeline.return_value = pipe

        result = limiter.check("t1")
        assert result.allowed is True
        assert result.limit == 100
        assert result.remaining == 49
