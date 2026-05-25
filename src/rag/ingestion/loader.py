from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

import fitz

from rag.models.document import RawDocument

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".pdf"}


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_json(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, str):
        return data
    return json.dumps(data, indent=2)


def _load_pdf(path: Path) -> str:
    doc = fitz.open(path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n\n".join(pages)


_LOADERS: dict[str, Callable[[Path], str]] = {
    ".txt": _load_text,
    ".md": _load_text,
    ".json": _load_json,
    ".pdf": _load_pdf,
}


def load_document(path: Path) -> RawDocument:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    loader = _LOADERS.get(suffix)
    if loader is None:
        raise ValueError(
            f"Unsupported file type: {suffix}. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    content = loader(path)
    logger.info("Loaded document", extra={"path": str(path), "chars": len(content)})

    return RawDocument(
        content=content,
        source_path=str(path),
        file_type=suffix,
    )
