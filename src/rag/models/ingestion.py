from dataclasses import dataclass


@dataclass(frozen=True)
class IngestionResult:
    documents_processed: int
    documents_skipped: int
    documents_failed: int
    chunks_created: int
    elapsed_ms: float
