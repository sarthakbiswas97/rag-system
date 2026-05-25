from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from rag.api.dependencies import get_generator, get_retriever, get_verification_pipeline
from rag.api.schemas import (
    CitationOut,
    QueryRequest,
    QueryResponse,
    TimingOut,
    VerificationOut,
)
from rag.generation.generator import Generator
from rag.retrieval.retriever import Retriever
from rag.tenancy.auth import get_current_tenant
from rag.tenancy.models import Tenant
from rag.verification.pipeline import VerificationPipeline

router = APIRouter()


@router.post("/v1/query", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    tenant: Tenant = Depends(get_current_tenant),
    retriever: Retriever = Depends(get_retriever),
    generator: Generator = Depends(get_generator),
    verification_pipeline: VerificationPipeline | None = Depends(
        get_verification_pipeline
    ),
) -> QueryResponse:
    start = time.perf_counter()

    retrieval_result = retriever.retrieve(
        body.question, top_k=body.top_k, tenant_id=tenant.id
    )
    generation_result = await generator.generate(
        body.question, retrieval_result, top_k=body.top_k
    )

    verification_out = None

    if verification_pipeline is not None:
        top_chunks = retrieval_result.scored_chunks[: body.top_k]
        verification_result = verification_pipeline.verify(
            generation_result, top_chunks
        )

        final_answer = verification_result.answer
        final_citations = verification_result.citations
        final_is_abstention = verification_result.is_abstention
        final_confidence = verification_result.confidence_score

        report = verification_result.verification_report
        cv = verification_result.citation_verifications
        ad = verification_result.abstention_decision

        verification_out = VerificationOut(
            faithfulness_score=(
                report.faithfulness_score if report else None
            ),
            citations_verified=len(cv) if cv else None,
            citations_supported=(
                sum(1 for v in cv if v.is_supported) if cv else None
            ),
            abstention_reason=(
                ad.reason if ad and ad.should_abstain else None
            ),
        )
    else:
        final_answer = generation_result.answer
        final_citations = generation_result.citations
        final_is_abstention = generation_result.is_abstention
        final_confidence = generation_result.confidence_score

    total_ms = (time.perf_counter() - start) * 1000

    citations = [
        CitationOut(
            index=i + 1,
            chunk_id=c.chunk_id,
            source=c.source_file,
            snippet=c.text_snippet,
        )
        for i, c in enumerate(final_citations)
    ]

    return QueryResponse(
        answer=final_answer,
        citations=citations,
        is_abstention=final_is_abstention,
        confidence=final_confidence,
        timing=TimingOut(
            retrieval_ms=round(retrieval_result.retrieval_time_ms, 1),
            generation_ms=round(generation_result.generation_time_ms, 1),
            total_ms=round(total_ms, 1),
        ),
        verification=verification_out,
    )
