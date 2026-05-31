from __future__ import annotations

import hashlib
import logging
import secrets

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag.tenancy.models import Tenant, TenantStatus

logger = logging.getLogger(__name__)

API_KEY_PREFIX = "rk_"
API_KEY_BYTE_LENGTH = 32


def generate_api_key() -> str:
    return f"{API_KEY_PREFIX}{secrets.token_hex(API_KEY_BYTE_LENGTH)}"


def hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


class TenantRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, name: str, email: str = "") -> tuple[Tenant, str]:
        api_key = generate_api_key()
        tenant = Tenant(
            name=name,
            email=email,
            api_key_hash=hash_api_key(api_key),
        )
        self._session.add(tenant)
        await self._session.commit()
        await self._session.refresh(tenant)
        logger.info(
            "Tenant created",
            extra={"tenant_id": tenant.id, "tenant_name": name},
        )
        return tenant, api_key

    async def get_by_id(self, tenant_id: str) -> Tenant | None:
        return await self._session.get(Tenant, tenant_id)

    async def get_by_api_key(self, api_key: str) -> Tenant | None:
        key_hash = hash_api_key(api_key)
        stmt = select(Tenant).where(
            Tenant.api_key_hash == key_hash,
            Tenant.status != TenantStatus.DELETED,
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_all(self, *, include_deleted: bool = False) -> list[Tenant]:
        stmt = select(Tenant).order_by(Tenant.created_at.desc())
        if not include_deleted:
            stmt = stmt.where(Tenant.status != TenantStatus.DELETED)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def update_status(
        self, tenant_id: str, status: TenantStatus
    ) -> Tenant | None:
        tenant = await self.get_by_id(tenant_id)
        if tenant is None:
            return None
        tenant.status = status
        await self._session.commit()
        await self._session.refresh(tenant)
        logger.info(
            "Tenant status updated",
            extra={"tenant_id": tenant_id, "status": status.value},
        )
        return tenant

    async def soft_delete(self, tenant_id: str) -> Tenant | None:
        return await self.update_status(tenant_id, TenantStatus.DELETED)

    async def update_name(self, tenant_id: str, name: str) -> Tenant | None:
        tenant = await self.get_by_id(tenant_id)
        if tenant is None:
            return None
        tenant.name = name
        await self._session.commit()
        await self._session.refresh(tenant)
        return tenant
