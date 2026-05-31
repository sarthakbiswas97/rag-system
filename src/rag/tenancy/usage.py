from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag.tenancy.models import EventType, UsageEvent

logger = logging.getLogger(__name__)


class UsageTracker:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record(
        self,
        tenant_id: str,
        event_type: EventType,
        value: int = 1,
        elapsed_ms: float | None = None,
    ) -> None:
        event = UsageEvent(
            tenant_id=tenant_id,
            event_type=event_type,
            value=value,
            elapsed_ms=elapsed_ms,
        )
        self._session.add(event)
        await self._session.commit()

    async def get_summary(
        self,
        tenant_id: str,
        since: datetime | None = None,
    ) -> dict[str, int]:
        stmt = (
            select(UsageEvent.event_type, func.sum(UsageEvent.value))
            .where(UsageEvent.tenant_id == tenant_id)
            .group_by(UsageEvent.event_type)
        )
        if since is not None:
            stmt = stmt.where(UsageEvent.created_at >= since)

        result = await self._session.execute(stmt)
        return {row[0]: int(row[1]) for row in result.all()}

    async def get_recent(
        self,
        tenant_id: str,
        limit: int = 20,
    ) -> list[UsageEvent]:
        stmt = (
            select(UsageEvent)
            .where(UsageEvent.tenant_id == tenant_id)
            .order_by(UsageEvent.created_at.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())
