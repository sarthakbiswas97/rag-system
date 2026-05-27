from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from rag.api.dependencies import get_db_session, get_vector_store
from rag.api.schemas import HealthResponse, LivenessResponse
from rag.retrieval.vector_store import VectorStore

router = APIRouter()


@router.get("/v1/health", response_model=HealthResponse)
async def health(
    request: Request,
    vector_store: VectorStore = Depends(get_vector_store),
    session: AsyncSession = Depends(get_db_session),
) -> HealthResponse:
    """Readiness probe -- checks all dependencies."""
    try:
        qdrant_ok = vector_store.collection_exists()
    except Exception:
        qdrant_ok = False

    try:
        await session.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    redis_ok = False
    redis_client = getattr(request.app.state, "_redis_client", None)
    if redis_client is not None:
        try:
            redis_ok = redis_client.ping()
        except Exception:
            redis_ok = False
    else:
        # No Redis configured -- don't mark as failure
        redis_ok = True

    all_ok = qdrant_ok and db_ok and redis_ok

    return HealthResponse(
        status="healthy" if all_ok else "degraded",
        qdrant="connected" if qdrant_ok else "unreachable",
        database="connected" if db_ok else "unreachable",
        redis="connected" if redis_ok else "unreachable",
    )


@router.get("/v1/livez", response_model=LivenessResponse)
async def liveness() -> LivenessResponse:
    """Liveness probe -- app is running, no dependency checks."""
    return LivenessResponse(status="alive")
