from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    session_id: str | None = None

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v.strip()


class CitationOut(BaseModel):
    index: int
    chunk_id: str
    source: str
    snippet: str


class TimingOut(BaseModel):
    retrieval_ms: float
    generation_ms: float
    total_ms: float


class VerificationOut(BaseModel):
    faithfulness_score: float | None = None
    citations_verified: int | None = None
    citations_supported: int | None = None
    abstention_reason: str | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    is_abstention: bool
    confidence: float
    timing: TimingOut
    verification: VerificationOut | None = None
    session_id: str | None = None


class IngestResponse(BaseModel):
    documents_processed: int
    documents_skipped: int
    documents_failed: int
    chunks_created: int
    elapsed_ms: float


class JobResponse(BaseModel):
    job_id: str
    status: str
    documents_processed: int = 0
    documents_skipped: int = 0
    documents_failed: int = 0
    chunks_created: int = 0
    elapsed_ms: float = 0.0
    error: str | None = None
    created_at: str
    completed_at: str | None = None


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    database: str
