from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from rag.api.dependencies import get_db_session
from rag.api.schemas_admin import (
    CreateTenantRequest,
    CreateTenantResponse,
    TenantListResponse,
    TenantOut,
)
from rag.tenancy.auth import require_admin
from rag.tenancy.models import Tenant
from rag.tenancy.repository import TenantRepository

router = APIRouter(prefix="/admin", dependencies=[Depends(require_admin)])


def _tenant_to_out(tenant: Tenant) -> TenantOut:
    return TenantOut(
        id=tenant.id,
        name=tenant.name,
        email=tenant.email,
        status=tenant.status.value,
        embedding_model_version=tenant.embedding_model_version,
        created_at=tenant.created_at.isoformat(),
        updated_at=tenant.updated_at.isoformat(),
    )


@router.post("/tenants", response_model=CreateTenantResponse, status_code=201)
async def create_tenant(
    body: CreateTenantRequest,
    session: AsyncSession = Depends(get_db_session),
) -> CreateTenantResponse:
    repo = TenantRepository(session)
    try:
        tenant, api_key = await repo.create(body.name, email=body.email)
    except IntegrityError as exc:
        raise HTTPException(
            status_code=409, detail="Tenant name already exists"
        ) from exc
    return CreateTenantResponse(tenant=_tenant_to_out(tenant), api_key=api_key)


@router.get("/tenants", response_model=TenantListResponse)
async def list_tenants(
    session: AsyncSession = Depends(get_db_session),
) -> TenantListResponse:
    repo = TenantRepository(session)
    tenants = await repo.list_all()
    out = [_tenant_to_out(t) for t in tenants]
    return TenantListResponse(tenants=out, count=len(out))


@router.get("/tenants/{tenant_id}", response_model=TenantOut)
async def get_tenant(
    tenant_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> TenantOut:
    repo = TenantRepository(session)
    tenant = await repo.get_by_id(tenant_id)
    if tenant is None:
        raise HTTPException(status_code=404, detail="Tenant not found")
    return _tenant_to_out(tenant)


@router.delete("/tenants/{tenant_id}", response_model=TenantOut)
async def delete_tenant(
    tenant_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> TenantOut:
    repo = TenantRepository(session)
    tenant = await repo.soft_delete(tenant_id)
    if tenant is None:
        raise HTTPException(status_code=404, detail="Tenant not found")
    return _tenant_to_out(tenant)
