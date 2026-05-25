from __future__ import annotations

from fastapi import APIRouter, Depends

from rag.api.dependencies import get_db_session, get_vector_store
from rag.api.schemas_admin import TenantOut, TenantStatsOut, UpdateTenantRequest
from rag.retrieval.vector_store import VectorStore
from rag.tenancy.auth import get_current_tenant
from rag.tenancy.models import Tenant
from rag.tenancy.repository import TenantRepository
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()


@router.get("/v1/me", response_model=TenantOut)
async def get_tenant_info(
    tenant: Tenant = Depends(get_current_tenant),
) -> TenantOut:
    return TenantOut(
        id=tenant.id,
        name=tenant.name,
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
        status=t.status.value,
        embedding_model_version=t.embedding_model_version,
        created_at=t.created_at.isoformat(),
        updated_at=t.updated_at.isoformat(),
    )
