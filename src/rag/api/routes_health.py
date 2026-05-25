from __future__ import annotations

from fastapi import APIRouter, Depends

from rag.api.dependencies import get_vector_store
from rag.api.schemas import HealthResponse
from rag.retrieval.vector_store import VectorStore

router = APIRouter()


@router.get("/v1/health", response_model=HealthResponse)
async def health(
    vector_store: VectorStore = Depends(get_vector_store),
) -> HealthResponse:
    try:
        qdrant_ok = vector_store.collection_exists()
    except Exception:
        qdrant_ok = False

    return HealthResponse(
        status="healthy" if qdrant_ok else "degraded",
        qdrant="connected" if qdrant_ok else "unreachable",
    )
