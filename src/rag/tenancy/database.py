from __future__ import annotations

import logging
import ssl
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from rag.tenancy.models import Base

logger = logging.getLogger(__name__)


def _prepare_database_url(database_url: str) -> tuple[str, dict]:
    """Normalize database URL and extract connect_args for asyncpg.

    Handles Neon/Supabase URLs that use sslmode=require which asyncpg
    does not understand as a query parameter.
    """
    connect_args: dict = {}

    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        return database_url, connect_args

    # Swap postgresql:// to postgresql+asyncpg:// if needed
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    # Strip query params that asyncpg doesn't understand
    parsed = urlparse(database_url)
    params = parse_qs(parsed.query)

    ssl_modes = ("require", "verify-full")
    needs_ssl = "sslmode" in params and params["sslmode"][0] in ssl_modes

    # Remove params asyncpg can't handle
    for key in ("sslmode", "channel_binding"):
        params.pop(key, None)

    cleaned_query = urlencode({k: v[0] for k, v in params.items()})
    database_url = urlunparse(parsed._replace(query=cleaned_query))

    if needs_ssl:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        connect_args["ssl"] = ssl_context

    return database_url, connect_args


def build_engine(
    database_url: str,
    pool_size: int = 5,
    max_overflow: int = 10,
    pool_recycle: int = 3600,
) -> AsyncEngine:
    database_url, connect_args = _prepare_database_url(database_url)
    kwargs: dict = {"echo": False}

    if not database_url.startswith("sqlite"):
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
