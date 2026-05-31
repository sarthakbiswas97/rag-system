from dataclasses import dataclass, field


@dataclass(frozen=True)
class DocumentInfo:
    """Metadata for a successfully processed document."""

    document_id: str
    source_file: str
    content_hash: str
    chunk_count: int


@dataclass(frozen=True)
class IngestionResult:
    documents_processed: int
    documents_skipped: int
    documents_failed: int
    chunks_created: int
    elapsed_ms: float
    processed_documents: tuple[DocumentInfo, ...] = field(default_factory=tuple)
