from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from redis import Redis

logger = logging.getLogger(__name__)

CACHE_PREFIX = "qcache:"
DEFAULT_TTL_SECONDS = 300  # 5 minutes


def _cache_key(tenant_id: str, query: str, top_k: int) -> str:
    raw = f"{tenant_id}:{query.strip().lower()}:{top_k}"
    digest = hashlib.sha256(raw.encode()).hexdigest()
    return f"{CACHE_PREFIX}{digest}"


class QueryCache:
    """Redis-backed cache for query responses.

    Caches serialized query results keyed by tenant + query + top_k.
    Supports tenant-scoped invalidation on ingestion.
    """

    def __init__(
        self,
        client: Redis,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        self._client = client
        self._ttl = ttl_seconds

    def get(
        self, tenant_id: str, query: str, top_k: int
    ) -> dict[str, Any] | None:
        key = _cache_key(tenant_id, query, top_k)
        raw = self._client.get(key)
        if raw is None:
            return None

        logger.debug(
            "Cache hit",
            extra={"tenant_id": tenant_id, "query": query[:80]},
        )
        return json.loads(raw)

    def set(
        self,
        tenant_id: str,
        query: str,
        top_k: int,
        response: dict[str, Any],
    ) -> None:
        key = _cache_key(tenant_id, query, top_k)
        self._client.setex(key, self._ttl, json.dumps(response))

        logger.debug(
            "Cache set",
            extra={"tenant_id": tenant_id, "query": query[:80]},
        )

    def invalidate_tenant(self, tenant_id: str) -> int:
        """Remove all cached queries for a tenant.

        Uses SCAN to avoid blocking Redis on large keyspaces.
        """
        # We cannot efficiently filter by tenant from key alone (SHA256),
        # so we store tenant_id in a Redis set for targeted invalidation.
        deleted = 0
        member_key = f"qcache_tenants:{tenant_id}"
        keys = self._client.smembers(member_key)

        if keys:
            deleted = self._client.delete(*keys)
            self._client.delete(member_key)

        logger.info(
            "Cache invalidated for tenant",
            extra={"tenant_id": tenant_id, "keys_deleted": deleted},
        )
        return deleted

    def set_with_tracking(
        self,
        tenant_id: str,
        query: str,
        top_k: int,
        response: dict[str, Any],
    ) -> None:
        """Set cache entry and track key for tenant-scoped invalidation."""
        key = _cache_key(tenant_id, query, top_k)
        member_key = f"qcache_tenants:{tenant_id}"

        pipe = self._client.pipeline()
        pipe.setex(key, self._ttl, json.dumps(response))
        pipe.sadd(member_key, key)
        pipe.expire(member_key, self._ttl * 2)
        pipe.execute()

        logger.debug(
            "Cache set with tracking",
            extra={"tenant_id": tenant_id, "query": query[:80]},
        )

    def clear_all(self) -> None:
        """Clear entire query cache. Use sparingly."""
        cursor = 0
        deleted = 0
        while True:
            cursor, keys = self._client.scan(
                cursor=cursor, match=f"{CACHE_PREFIX}*", count=100
            )
            if keys:
                self._client.delete(*keys)
                deleted += len(keys)
            if cursor == 0:
                break

        logger.info("Cache cleared", extra={"keys_deleted": deleted})
