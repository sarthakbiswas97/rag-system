from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from rag.api.dependencies import get_db_session
from rag.api.schemas_admin import CreateTenantResponse, RegisterRequest, TenantOut
from rag.tenancy.models import Tenant
from rag.tenancy.repository import TenantRepository

router = APIRouter()


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


@router.post("/v1/register", response_model=CreateTenantResponse, status_code=201)
async def register(
    body: RegisterRequest,
    session: AsyncSession = Depends(get_db_session),
) -> CreateTenantResponse:
    repo = TenantRepository(session)
    try:
        tenant, api_key = await repo.create(body.name, email=body.email)
    except IntegrityError as exc:
        raise HTTPException(
            status_code=409, detail="Name or email already registered"
        ) from exc
    return CreateTenantResponse(tenant=_tenant_to_out(tenant), api_key=api_key)
