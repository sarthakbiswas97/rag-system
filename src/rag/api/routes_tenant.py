from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from rag.api.dependencies import get_db_session, get_vector_store
from rag.api.schemas_admin import (
    TenantOut,
    TenantStatsOut,
    UpdateTenantRequest,
    UsageEventOut,
    UsageSummaryOut,
)
from rag.retrieval.vector_store import VectorStore
from rag.tenancy.auth import get_current_tenant
from rag.tenancy.models import Tenant
from rag.tenancy.repository import TenantRepository
from rag.tenancy.usage import UsageTracker

router = APIRouter()


@router.get("/v1/me", response_model=TenantOut)
async def get_tenant_info(
    tenant: Tenant = Depends(get_current_tenant),
) -> TenantOut:
    return TenantOut(
        id=tenant.id,
        name=tenant.name,
        email=tenant.email,
        status=tenant.status.value,
        embedding_model_version=tenant.embedding_model_version,
        created_at=tenant.created_at.isoformat(),
        updated_at=tenant.updated_at.isoformat(),
    )


@router.get("/v1/me/stats", response_model=TenantStatsOut)
async def get_tenant_stats(
    tenant: Tenant = Depends(get_current_tenant),
    vector_store: VectorStore = Depends(get_vector_store),
) -> TenantStatsOut:
    chunk_count = vector_store.count_by_tenant(tenant.id)
    return TenantStatsOut(
        tenant_id=tenant.id,
        chunk_count=chunk_count,
    )


@router.put("/v1/me", response_model=TenantOut)
async def update_tenant_info(
    body: UpdateTenantRequest,
    tenant: Tenant = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> TenantOut:
    repo = TenantRepository(session)
    updated = await repo.update_name(tenant.id, body.name)
    t = updated if updated else tenant
    return TenantOut(
        id=t.id,
        name=t.name,
        email=t.email,
        status=t.status.value,
        embedding_model_version=t.embedding_model_version,
        created_at=t.created_at.isoformat(),
        updated_at=t.updated_at.isoformat(),
    )


@router.get("/v1/me/usage", response_model=UsageSummaryOut)
async def get_tenant_usage(
    tenant: Tenant = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
    days: int = Query(default=30, ge=1, le=365),
) -> UsageSummaryOut:
    tracker = UsageTracker(session)
    since = datetime.now(tz=UTC) - timedelta(days=days)
    totals = await tracker.get_summary(tenant.id, since=since)
    recent = await tracker.get_recent(tenant.id, limit=20)
    return UsageSummaryOut(
        tenant_id=tenant.id,
        period=f"last_{days}_days",
        totals=totals,
        recent=[
            UsageEventOut(
                event_type=e.event_type.value,
                value=e.value,
                elapsed_ms=e.elapsed_ms,
                created_at=e.created_at.isoformat(),
            )
            for e in recent
        ],
    )
