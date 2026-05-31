from __future__ import annotations

import json
import time

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from rag.api.dependencies import (
    get_db_session,
    get_generator,
    get_llm_client,
    get_query_cache,
    get_query_rate_limiter,
    get_retriever,
    get_session_store,
    get_verification_pipeline,
)
from rag.api.rate_limiter import RateLimiter, RateLimitResult
from rag.api.schemas import (
    CitationOut,
    QueryRequest,
    QueryResponse,
    TimingOut,
    VerificationOut,
)
from rag.generation.generator import (
    Generator,
    compute_confidence,
    detect_abstention,
    parse_citations,
)
from rag.generation.llm_client import LLMClient, LLMServiceError
from rag.generation.prompt_builder import build_rag_prompt
from rag.models.generation import GenerationResponse
from rag.retrieval.cache import QueryCache
from rag.retrieval.conversational_rewriter import rewrite_with_context
from rag.retrieval.retriever import Retriever
from rag.session.store import SessionStore
from rag.tenancy.auth import get_current_tenant
from rag.tenancy.models import EventType, Tenant
from rag.tenancy.usage import UsageTracker
from rag.verification.pipeline import VerificationPipeline

router = APIRouter()


def _add_rate_limit_headers(response: Response, result: RateLimitResult) -> None:
    response.headers["X-RateLimit-Limit"] = str(result.limit)
    response.headers["X-RateLimit-Remaining"] = str(result.remaining)
    response.headers["X-RateLimit-Reset"] = str(result.reset_after)


@router.post("/v1/query", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    response: Response,
    tenant: Tenant = Depends(get_current_tenant),
    retriever: Retriever = Depends(get_retriever),
    generator: Generator = Depends(get_generator),
    llm_client: LLMClient = Depends(get_llm_client),
    verification_pipeline: VerificationPipeline | None = Depends(
        get_verification_pipeline
    ),
    session_store: SessionStore | None = Depends(get_session_store),
    query_cache: QueryCache | None = Depends(get_query_cache),
    rate_limiter: RateLimiter | None = Depends(get_query_rate_limiter),
    session_db: AsyncSession = Depends(get_db_session),
) -> QueryResponse:
    if rate_limiter is not None:
        result = rate_limiter.check(tenant.id, "query")
        _add_rate_limit_headers(response, result)
        if not result.allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(result.reset_after)},
            )

    start = time.perf_counter()

    session_id = body.session_id
    session = None

    if session_store is not None and session_id is not None:
        session = session_store.get(session_id)

    if session_store is not None and session is None and session_id is not None:
        session = session_store.create(tenant_id=tenant.id)
        session_id = session.session_id

    if session is not None and session_store is not None:
        session = session_store.add_turn(session.session_id, "user", body.question)

    # Check cache for non-conversational queries (no session context)
    is_cacheable = session is None and query_cache is not None
    if is_cacheable:
        cached = query_cache.get(tenant.id, body.question, body.top_k)
        if cached is not None:
            return QueryResponse(**cached)

    search_query = await rewrite_with_context(body.question, session, llm_client)

    retrieval_result = retriever.retrieve(
        search_query, top_k=body.top_k, tenant_id=tenant.id
    )
    try:
        generation_result = await generator.generate(
            body.question, retrieval_result, top_k=body.top_k
        )
    except LLMServiceError:
        raise HTTPException(
            status_code=503,
            detail="LLM service temporarily unavailable. Please try again later.",
        ) from None

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
            faithfulness_score=(report.faithfulness_score if report else None),
            citations_verified=len(cv) if cv else None,
            citations_supported=(sum(1 for v in cv if v.is_supported) if cv else None),
            abstention_reason=(ad.reason if ad and ad.should_abstain else None),
        )
    else:
        final_answer = generation_result.answer
        final_citations = generation_result.citations
        final_is_abstention = generation_result.is_abstention
        final_confidence = generation_result.confidence_score

    if session is not None and session_store is not None:
        session_store.add_turn(session.session_id, "assistant", final_answer)

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

    response = QueryResponse(
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
        session_id=session_id if session is not None else None,
    )

    if is_cacheable:
        query_cache.set_with_tracking(
            tenant.id, body.question, body.top_k, response.model_dump()
        )

    tracker = UsageTracker(session_db)
    await tracker.record(
        tenant_id=tenant.id,
        event_type=EventType.QUERY,
        elapsed_ms=round(total_ms, 1),
    )

    return response


@router.post("/v1/query/stream")
async def query_stream(
    body: QueryRequest,
    response: Response,
    tenant: Tenant = Depends(get_current_tenant),
    retriever: Retriever = Depends(get_retriever),
    llm_client: LLMClient = Depends(get_llm_client),
    verification_pipeline: VerificationPipeline | None = Depends(
        get_verification_pipeline
    ),
    session_store: SessionStore | None = Depends(get_session_store),
    rate_limiter: RateLimiter | None = Depends(get_query_rate_limiter),
    session_db: AsyncSession = Depends(get_db_session),
):
    if rate_limiter is not None:
        result = rate_limiter.check(tenant.id, "query")
        _add_rate_limit_headers(response, result)
        if not result.allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(result.reset_after)},
            )

    async def _stream():
        start = time.perf_counter()

        session = None
        session_id = body.session_id

        if session_store is not None and session_id is not None:
            session = session_store.get(session_id)

        if session_store is not None and session is None and session_id is not None:
            session = session_store.create(tenant_id=tenant.id)
            session_id = session.session_id

        if session is not None and session_store is not None:
            session = session_store.add_turn(session.session_id, "user", body.question)

        search_query = await rewrite_with_context(body.question, session, llm_client)

        retrieval_result = retriever.retrieve(
            search_query, top_k=body.top_k, tenant_id=tenant.id
        )

        top_chunks = retrieval_result.scored_chunks[: body.top_k]
        system_prompt, user_prompt = build_rag_prompt(body.question, top_chunks)

        # Stream tokens
        accumulated: list[str] = []
        try:
            async for item in llm_client.stream_generate(system_prompt, user_prompt):
                if isinstance(item, str):
                    accumulated.append(item)
                    yield f"data: {json.dumps({'type': 'token', 'content': item})}\n\n"
                # The last item is the LLMResponse, skip yielding it
        except LLMServiceError as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
            return

        full_answer = "".join(accumulated)

        # Run verification
        citations = parse_citations(full_answer, top_chunks)
        is_abstention = detect_abstention(full_answer, has_citations=len(citations) > 0)
        confidence = compute_confidence(top_chunks)

        gen_response = GenerationResponse(
            answer=full_answer,
            citations=citations,
            is_abstention=is_abstention,
            confidence_score=confidence,
            retrieval_result=retrieval_result,
            generation_time_ms=0.0,
        )

        final_answer = full_answer
        final_citations = citations
        final_is_abstention = is_abstention
        verification_out = None
        abstention_reason = None

        if verification_pipeline is not None:
            v_result = verification_pipeline.verify(gen_response, top_chunks)
            final_answer = v_result.answer
            final_citations = v_result.citations
            final_is_abstention = v_result.is_abstention

            report = v_result.verification_report
            ad = v_result.abstention_decision
            verification_out = VerificationOut(
                faithfulness_score=(report.faithfulness_score if report else None),
                citations_verified=len(v_result.citation_verifications)
                if v_result.citation_verifications
                else None,
                citations_supported=(
                    sum(1 for v in v_result.citation_verifications if v.is_supported)
                    if v_result.citation_verifications
                    else None
                ),
                abstention_reason=(ad.reason if ad and ad.should_abstain else None),
            )
            if ad and ad.should_abstain:
                abstention_reason = ad.reason

        if session is not None and session_store is not None:
            session_store.add_turn(session.session_id, "assistant", final_answer)

        total_ms = (time.perf_counter() - start) * 1000

        citations_out = [
            {
                "index": i + 1,
                "chunk_id": c.chunk_id,
                "source": c.source_file,
                "snippet": c.text_snippet,
            }
            for i, c in enumerate(final_citations)
        ]

        if final_is_abstention:
            payload = {
                "type": "abstention",
                "answer": final_answer,
                "reason": abstention_reason,
            }
            yield f"data: {json.dumps(payload)}\n\n"
        else:
            payload = {
                "type": "done",
                "answer": final_answer,
                "citations": citations_out,
                "verification": verification_out.model_dump()
                if verification_out
                else None,
                "confidence": confidence,
                "timing": {
                    "retrieval_ms": round(retrieval_result.retrieval_time_ms, 1),
                    "total_ms": round(total_ms, 1),
                },
            }
            yield f"data: {json.dumps(payload)}\n\n"

        # Record usage
        tracker = UsageTracker(session_db)
        await tracker.record(
            tenant_id=tenant.id,
            event_type=EventType.QUERY,
            elapsed_ms=round(total_ms, 1),
        )

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
