from __future__ import annotations

import hashlib
import re

from rag.models.document import RawDocument


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_hash(text: str) -> str:
    return hashlib.sha256(_normalize(text).encode("utf-8")).hexdigest()


class ContentHasher:
    def __init__(self) -> None:
        self._seen: set[str] = set()

    def is_duplicate(self, doc: RawDocument, tenant_id: str = "") -> bool:
        doc_hash = compute_hash(doc.content)
        scoped_key = f"{tenant_id}:{doc_hash}"
        if scoped_key in self._seen:
            return True
        self._seen.add(scoped_key)
        return False

    def reset(self) -> None:
        self._seen.clear()

    @property
    def seen_count(self) -> int:
        return len(self._seen)
