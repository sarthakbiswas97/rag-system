from __future__ import annotations

import logging
from collections.abc import Sequence

from rag.models.document import Chunk, ChunkMetadata, RawDocument

logger = logging.getLogger(__name__)

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _token_count(text: str) -> int:
    """Approximate token count via whitespace splitting."""
    return len(text.split())


def _hard_split(text: str, chunk_size: int) -> list[str]:
    """Last-resort character-level split when no separators work."""
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        if current_len + 1 > chunk_size and current:
            chunks.append(" ".join(current))
            current = []
            current_len = 0
        current.append(word)
        current_len += 1

    if current:
        chunks.append(" ".join(current))
    return chunks


def _merge_splits(
    splits: Sequence[str],
    separator: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Merge small text splits into chunks that fit within chunk_size tokens."""
    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for split in splits:
        split_tokens = _token_count(split)

        if current_tokens + split_tokens > chunk_size and current_parts:
            chunks.append(separator.join(current_parts))

            # apply overlap: keep trailing parts from previous chunk
            overlap_parts: list[str] = []
            overlap_tokens = 0
            for part in reversed(current_parts):
                part_tokens = _token_count(part)
                if overlap_tokens + part_tokens > chunk_overlap:
                    break
                overlap_parts.append(part)
                overlap_tokens += part_tokens

            current_parts = list(reversed(overlap_parts))
            current_tokens = overlap_tokens

        current_parts.append(split)
        current_tokens += split_tokens

    if current_parts:
        chunks.append(separator.join(current_parts))

    return chunks


def _split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str],
) -> list[str]:
    """Recursively split text using a hierarchy of separators."""
    if not text.strip():
        return []

    if _token_count(text) <= chunk_size:
        return [text.strip()]

    # find first applicable separator
    chosen_sep = ""
    chosen_idx = len(separators) - 1
    for i, sep in enumerate(separators):
        if sep == "":
            chosen_sep = sep
            chosen_idx = i
            break
        if sep in text:
            chosen_sep = sep
            chosen_idx = i
            break

    # hard split if we've exhausted all separators
    if chosen_sep == "":
        return _hard_split(text, chunk_size)

    # split on chosen separator
    raw_splits = text.split(chosen_sep)
    splits = [s for s in raw_splits if s.strip()]

    # merge small splits into chunks
    merged = _merge_splits(splits, chosen_sep, chunk_size, chunk_overlap)

    # recursively split any chunks that are still too large
    remaining_seps = separators[chosen_idx + 1 :]
    result: list[str] = []
    for chunk in merged:
        if _token_count(chunk) > chunk_size and remaining_seps:
            result.extend(_split_text(chunk, chunk_size, chunk_overlap, remaining_seps))
        else:
            stripped = chunk.strip()
            if stripped:
                result.append(stripped)

    return result


def chunk_document(
    doc: RawDocument,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    separators: list[str] | None = None,
    tenant_id: str = "",
) -> tuple[Chunk, ...]:
    """Split a RawDocument into a tuple of frozen Chunks."""
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than "
            f"chunk_size ({chunk_size})"
        )

    if not doc.content.strip():
        return ()

    seps = separators if separators is not None else DEFAULT_SEPARATORS
    texts = _split_text(doc.content, chunk_size, chunk_overlap, seps)
    total = len(texts)

    chunks = tuple(
        Chunk(
            document_id=doc.document_id,
            text=text,
            metadata=ChunkMetadata(
                source_file=doc.source_path,
                tenant_id=tenant_id,
                chunk_index=i,
                total_chunks=total,
            ),
        )
        for i, text in enumerate(texts)
    )

    logger.info(
        "Chunked document",
        extra={
            "document_id": doc.document_id,
            "source": doc.source_path,
            "chunks": total,
            "chunk_size": chunk_size,
            "overlap": chunk_overlap,
        },
    )

    return chunks
