from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from rag.api.dependencies import get_generator, get_retriever
from rag.api.schemas import CitationOut, QueryRequest, QueryResponse, TimingOut
from rag.generation.generator import Generator
from rag.retrieval.retriever import Retriever

router = APIRouter()


@router.post("/v1/query", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    retriever: Retriever = Depends(get_retriever),
    generator: Generator = Depends(get_generator),
) -> QueryResponse:
    start = time.perf_counter()

    retrieval_result = retriever.retrieve(body.question, top_k=body.top_k)
    generation_result = await generator.generate(
        body.question, retrieval_result, top_k=body.top_k
    )

    total_ms = (time.perf_counter() - start) * 1000

    citations = [
        CitationOut(
            index=i + 1,
            chunk_id=c.chunk_id,
            source=c.source_file,
            snippet=c.text_snippet,
        )
        for i, c in enumerate(generation_result.citations)
    ]

    return QueryResponse(
        answer=generation_result.answer,
        citations=citations,
        is_abstention=generation_result.is_abstention,
        confidence=generation_result.confidence_score,
        timing=TimingOut(
            retrieval_ms=round(retrieval_result.retrieval_time_ms, 1),
            generation_ms=round(generation_result.generation_time_ms, 1),
            total_ms=round(total_ms, 1),
        ),
    )
