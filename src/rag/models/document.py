from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

from qdrant_client.models import SparseVector


@dataclass(frozen=True)
class ChunkMetadata:
    source_file: str
    tenant_id: str = ""
    page_number: int | None = None
    section_title: str | None = None
    chunk_index: int = 0
    total_chunks: int = 0
    parent_chunk_id: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())


@dataclass(frozen=True)
class Chunk:
    chunk_id: str = field(default_factory=lambda: str(uuid4()))
    document_id: str = ""
    text: str = ""
    metadata: ChunkMetadata = field(
        default_factory=lambda: ChunkMetadata(source_file="")
    )
    embedding: tuple[float, ...] | None = None
    sparse_embedding: SparseVector | None = None


@dataclass(frozen=True)
class RawDocument:
    document_id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    source_path: str = ""
    file_type: str = ""
