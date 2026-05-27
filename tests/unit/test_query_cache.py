from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rag.retrieval.cache import (
    CACHE_PREFIX,
    DEFAULT_TTL_SECONDS,
    QueryCache,
    _cache_key,
)


@pytest.fixture()
def mock_redis() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def cache(mock_redis: MagicMock) -> QueryCache:
    return QueryCache(client=mock_redis)


class TestCacheKey:
    def test_deterministic(self) -> None:
        k1 = _cache_key("t1", "hello", 5)
        k2 = _cache_key("t1", "hello", 5)
        assert k1 == k2

    def test_has_prefix(self) -> None:
        key = _cache_key("t1", "hello", 5)
        assert key.startswith(CACHE_PREFIX)

    def test_different_tenants_produce_different_keys(self) -> None:
        k1 = _cache_key("t1", "hello", 5)
        k2 = _cache_key("t2", "hello", 5)
        assert k1 != k2

    def test_different_queries_produce_different_keys(self) -> None:
        k1 = _cache_key("t1", "hello", 5)
        k2 = _cache_key("t1", "world", 5)
        assert k1 != k2

    def test_different_top_k_produce_different_keys(self) -> None:
        k1 = _cache_key("t1", "hello", 5)
        k2 = _cache_key("t1", "hello", 10)
        assert k1 != k2

    def test_normalizes_case_and_whitespace(self) -> None:
        k1 = _cache_key("t1", "Hello World", 5)
        k2 = _cache_key("t1", "  hello world  ", 5)
        assert k1 == k2


class TestQueryCacheGet:
    def test_cache_miss_returns_none(
        self, cache: QueryCache, mock_redis: MagicMock
    ) -> None:
        mock_redis.get.return_value = None
        result = cache.get("t1", "hello", 5)
        assert result is None

    def test_cache_hit_returns_dict(
        self, cache: QueryCache, mock_redis: MagicMock
    ) -> None:
        import json

        data = {"answer": "Paris", "confidence": 0.9}
        mock_redis.get.return_value = json.dumps(data).encode()
        result = cache.get("t1", "hello", 5)
        assert result == data


class TestQueryCacheSet:
    def test_set_calls_redis_setex(
        self, cache: QueryCache, mock_redis: MagicMock
    ) -> None:
        cache.set("t1", "hello", 5, {"answer": "Paris"})
        mock_redis.setex.assert_called_once()
        args = mock_redis.setex.call_args
        assert args[0][1] == DEFAULT_TTL_SECONDS

    def test_set_with_tracking_uses_pipeline(
        self, cache: QueryCache, mock_redis: MagicMock
    ) -> None:
        pipe = MagicMock()
        mock_redis.pipeline.return_value = pipe
        cache.set_with_tracking("t1", "hello", 5, {"answer": "Paris"})
        pipe.setex.assert_called_once()
        pipe.sadd.assert_called_once()
        pipe.expire.assert_called_once()
        pipe.execute.assert_called_once()


class TestQueryCacheInvalidation:
    def test_invalidate_tenant_deletes_tracked_keys(
        self, cache: QueryCache, mock_redis: MagicMock
    ) -> None:
        mock_redis.smembers.return_value = {b"key1", b"key2"}
        mock_redis.delete.return_value = 2
        deleted = cache.invalidate_tenant("t1")
        assert deleted == 2

    def test_invalidate_tenant_no_keys(
        self, cache: QueryCache, mock_redis: MagicMock
    ) -> None:
        mock_redis.smembers.return_value = set()
        deleted = cache.invalidate_tenant("t1")
        assert deleted == 0

    def test_clear_all_scans_and_deletes(
        self, cache: QueryCache, mock_redis: MagicMock
    ) -> None:
        mock_redis.scan.return_value = (0, [b"qcache:abc", b"qcache:def"])
        mock_redis.delete.return_value = 2
        cache.clear_all()
        mock_redis.delete.assert_called_once()


class TestQueryCacheCustomTTL:
    def test_custom_ttl(self, mock_redis: MagicMock) -> None:
        cache = QueryCache(client=mock_redis, ttl_seconds=60)
        cache.set("t1", "hello", 5, {"answer": "Paris"})
        args = mock_redis.setex.call_args
        assert args[0][1] == 60
