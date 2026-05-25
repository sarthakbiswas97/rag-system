from dataclasses import dataclass

from rag.models.retrieval import RetrievalResult


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    source_file: str
    text_snippet: str
    sentence_index: int


@dataclass(frozen=True)
class GenerationResponse:
    answer: str
    citations: tuple[Citation, ...]
    is_abstention: bool
    confidence_score: float
    retrieval_result: RetrievalResult
    generation_time_ms: float
