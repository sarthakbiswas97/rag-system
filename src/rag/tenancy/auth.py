from __future__ import annotations

import logging

from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from rag.api.dependencies import get_db_session
from rag.tenancy.models import Tenant, TenantStatus
from rag.tenancy.repository import TenantRepository

logger = logging.getLogger(__name__)


async def get_current_tenant(
    x_api_key: str = Header(..., description="Tenant API key"),
    session: AsyncSession = Depends(get_db_session),
) -> Tenant:
    repo = TenantRepository(session)
    tenant = await repo.get_by_api_key(x_api_key)

    if tenant is None:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if tenant.status == TenantStatus.SUSPENDED:
        raise HTTPException(status_code=403, detail="Tenant account is suspended")

    return tenant


async def require_admin(
    request: Request,
    x_api_key: str = Header(..., description="Admin API key"),
) -> None:
    admin_key = request.app.state.admin_api_key
    if not admin_key:
        raise HTTPException(status_code=503, detail="Admin access not configured")

    if x_api_key != admin_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")
