from dataclasses import dataclass


@dataclass(frozen=True)
class EntailmentResult:
    sentence: str
    label: str  # "entailment" | "contradiction" | "neutral"
    confidence: float
    supporting_chunk_id: str | None


@dataclass(frozen=True)
class VerificationReport:
    sentence_results: tuple[EntailmentResult, ...]
    overall_faithful: bool
    faithfulness_score: float
