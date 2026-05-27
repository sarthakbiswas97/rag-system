from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from rag.tenancy.models import Base

logger = logging.getLogger(__name__)


def build_engine(
    database_url: str,
    pool_size: int = 5,
    max_overflow: int = 10,
    pool_recycle: int = 3600,
) -> AsyncEngine:
    connect_args = {}
    kwargs: dict = {"echo": False}

    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    else:
        # Connection pool tuning for PostgreSQL / MySQL
        kwargs["pool_size"] = pool_size
        kwargs["max_overflow"] = max_overflow
        kwargs["pool_recycle"] = pool_recycle
        kwargs["pool_pre_ping"] = True

    return create_async_engine(
        database_url,
        connect_args=connect_args,
        **kwargs,
    )


def build_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_db(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialized")


async def close_db(engine: AsyncEngine) -> None:
    await engine.dispose()
    logger.info("Database engine disposed")
