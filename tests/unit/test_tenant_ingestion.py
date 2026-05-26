from __future__ import annotations

from rag.ingestion.chunker import chunk_document
from rag.ingestion.hasher import ContentHasher
from rag.models.document import RawDocument


class TestChunkerTenantId:
    def test_chunks_carry_tenant_id(self) -> None:
        doc = RawDocument(content="Hello world. This is a test.", source_path="a.txt")
        chunks = chunk_document(doc, chunk_size=100, chunk_overlap=0, tenant_id="t-1")
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.metadata.tenant_id == "t-1"

    def test_default_tenant_id_is_empty(self) -> None:
        doc = RawDocument(content="Hello world.", source_path="a.txt")
        chunks = chunk_document(doc, chunk_size=100, chunk_overlap=0)
        assert chunks[0].metadata.tenant_id == ""


class TestHasherTenantScoped:
    def test_same_content_different_tenants_not_duplicate(self) -> None:
        hasher = ContentHasher()
        doc = RawDocument(content="same content", source_path="a.txt")

        assert hasher.is_duplicate(doc, tenant_id="t-a") is False
        assert hasher.is_duplicate(doc, tenant_id="t-b") is False

    def test_same_content_same_tenant_is_duplicate(self) -> None:
        hasher = ContentHasher()
        doc = RawDocument(content="same content", source_path="a.txt")

        assert hasher.is_duplicate(doc, tenant_id="t-a") is False
        assert hasher.is_duplicate(doc, tenant_id="t-a") is True

    def test_backward_compatible_no_tenant(self) -> None:
        hasher = ContentHasher()
        doc = RawDocument(content="content", source_path="a.txt")

        assert hasher.is_duplicate(doc) is False
        assert hasher.is_duplicate(doc) is True
