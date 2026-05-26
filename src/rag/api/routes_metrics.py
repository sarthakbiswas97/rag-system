from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest

router = APIRouter()


@router.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    return generate_latest().decode("utf-8")
