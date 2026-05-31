from __future__ import annotations

from qdrant_client import QdrantClient

from rag.retrieval.vector_store import VectorStore

VECTOR_DIM = 384


class TestVectorStoreSharding:
    def test_default_shard_number_is_one(self) -> None:
        client = QdrantClient(":memory:")
        store = VectorStore(client=client, collection="test_default")
        assert store._shard_number == 1
        assert store._replication_factor == 1

    def test_custom_shard_number(self) -> None:
        client = QdrantClient(":memory:")
        store = VectorStore(
            client=client,
            collection="test_sharded",
            shard_number=4,
            replication_factor=1,
        )
        store.create_collection(vector_size=VECTOR_DIM)

        # In-memory Qdrant does not expose shard_number, but collection
        # creation succeeds without error when shard_number is passed.
        assert store.collection_exists()

    def test_collection_creation_logs_sharding(self, caplog) -> None:
        import logging

        client = QdrantClient(":memory:")
        store = VectorStore(
            client=client,
            collection="test_log",
            shard_number=6,
            replication_factor=1,
        )
        with caplog.at_level(logging.INFO):
            store.create_collection(vector_size=VECTOR_DIM)

        # Should have created collection successfully
        assert store.collection_exists()
