from dataclasses import dataclass

from rag.models.document import Chunk


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    retrieval_method: str


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    rewritten_query: str | None
    scored_chunks: tuple[ScoredChunk, ...]
    retrieval_time_ms: float
