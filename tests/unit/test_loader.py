from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.ingestion.loader import SUPPORTED_EXTENSIONS, load_document


@pytest.fixture()
def tmp_text_file(tmp_path: Path) -> Path:
    path = tmp_path / "sample.txt"
    path.write_text("Hello, world!", encoding="utf-8")
    return path


@pytest.fixture()
def tmp_md_file(tmp_path: Path) -> Path:
    path = tmp_path / "readme.md"
    path.write_text("# Title\n\nSome content.", encoding="utf-8")
    return path


@pytest.fixture()
def tmp_json_file(tmp_path: Path) -> Path:
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"key": "value", "nested": [1, 2]}), encoding="utf-8")
    return path


@pytest.fixture()
def tmp_json_string_file(tmp_path: Path) -> Path:
    path = tmp_path / "plain.json"
    path.write_text(json.dumps("just a string"), encoding="utf-8")
    return path


class TestLoadDocument:
    def test_loads_txt(self, tmp_text_file: Path) -> None:
        doc = load_document(tmp_text_file)
        assert doc.content == "Hello, world!"
        assert doc.file_type == ".txt"
        assert doc.source_path == str(tmp_text_file)
        assert doc.document_id  # non-empty uuid

    def test_loads_md(self, tmp_md_file: Path) -> None:
        doc = load_document(tmp_md_file)
        assert "# Title" in doc.content
        assert doc.file_type == ".md"

    def test_loads_json_object(self, tmp_json_file: Path) -> None:
        doc = load_document(tmp_json_file)
        assert "key" in doc.content
        assert "value" in doc.content
        assert doc.file_type == ".json"

    def test_loads_json_string(self, tmp_json_string_file: Path) -> None:
        doc = load_document(tmp_json_string_file)
        assert doc.content == "just a string"

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.txt"
        path.write_text("", encoding="utf-8")
        doc = load_document(path)
        assert doc.content == ""

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "nonexistent.txt"
        with pytest.raises(FileNotFoundError, match="File not found"):
            load_document(path)

    def test_raises_on_unsupported_extension(self, tmp_path: Path) -> None:
        path = tmp_path / "data.csv"
        path.write_text("a,b,c", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_document(path)

    def test_unique_ids_per_load(self, tmp_text_file: Path) -> None:
        doc_a = load_document(tmp_text_file)
        doc_b = load_document(tmp_text_file)
        assert doc_a.document_id != doc_b.document_id

    def test_supported_extensions_constant(self) -> None:
        assert ".txt" in SUPPORTED_EXTENSIONS
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".md" in SUPPORTED_EXTENSIONS
        assert ".json" in SUPPORTED_EXTENSIONS
