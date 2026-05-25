from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from rag.api.dependencies import get_db_session, get_vector_store
from rag.api.schemas import HealthResponse
from rag.retrieval.vector_store import VectorStore

router = APIRouter()


@router.get("/v1/health", response_model=HealthResponse)
async def health(
    vector_store: VectorStore = Depends(get_vector_store),
    session: AsyncSession = Depends(get_db_session),
) -> HealthResponse:
    try:
        qdrant_ok = vector_store.collection_exists()
    except Exception:
        qdrant_ok = False

    try:
        await session.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    all_ok = qdrant_ok and db_ok

    return HealthResponse(
        status="healthy" if all_ok else "degraded",
        qdrant="connected" if qdrant_ok else "unreachable",
        database="connected" if db_ok else "unreachable",
    )
