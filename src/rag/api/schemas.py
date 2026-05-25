from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)

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


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    is_abstention: bool
    confidence: float
    timing: TimingOut


class IngestResponse(BaseModel):
    documents_processed: int
    documents_skipped: int
    documents_failed: int
    chunks_created: int
    elapsed_ms: float


class HealthResponse(BaseModel):
    status: str
    qdrant: str
