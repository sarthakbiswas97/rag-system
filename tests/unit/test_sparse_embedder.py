from __future__ import annotations

import math

from rag.retrieval.sparse_embedder import SparseEmbedder


class TestSparseEmbedder:
    def test_empty_text_returns_empty_embedding(self) -> None:
        embedder = SparseEmbedder()
        result = embedder.embed_text("")
        assert result.indices == ()
        assert result.values == ()

    def test_tokenizes_and_counts(self) -> None:
        embedder = SparseEmbedder()
        result = embedder.embed_text("hello world hello")

        assert len(result.indices) == 2  # "hello", "world"
        assert len(result.values) == 2

        # Values are raw counts
        values_dict = dict(zip(result.indices, result.values, strict=False))
        assert 2.0 in values_dict.values()  # "hello" appears twice
        assert 1.0 in values_dict.values()  # "world" appears once

    def test_indices_are_sorted(self) -> None:
        embedder = SparseEmbedder()
        result = embedder.embed_text("zebra apple banana")
        assert list(result.indices) == sorted(result.indices)

    def test_vocab_grows_incrementally(self) -> None:
        embedder = SparseEmbedder()
        embedder.embed_text("cat dog")
        assert embedder.vocab_size == 2

        embedder.embed_text("cat bird")
        assert embedder.vocab_size == 3

    def test_vocab_size_limit(self) -> None:
        embedder = SparseEmbedder(max_vocab_size=2)
        embedder.embed_text("one two three")
        assert embedder.vocab_size == 2

    def test_repeated_text_uses_same_indices(self) -> None:
        embedder = SparseEmbedder()
        r1 = embedder.embed_text("hello world")
        r2 = embedder.embed_text("hello world")

        assert r1.indices == r2.indices
        assert r1.values == r2.values

    def test_query_sublinear_scaling(self) -> None:
        embedder = SparseEmbedder()
        embedder.embed_text("hello hello hello")  # Build vocab

        query = embedder.embed_query("hello hello")
        values_dict = dict(zip(query.indices, query.values, strict=False))

        # value = 1 + log(tf) = 1 + log(2) ≈ 1.693
        expected = 1.0 + math.log(2)
        assert math.isclose(values_dict[query.indices[0]], expected, rel_tol=1e-9)

    def test_query_ignores_unknown_tokens(self) -> None:
        embedder = SparseEmbedder()
        embedder.embed_text("known word")

        query = embedder.embed_query("unknown xyz")
        assert query.indices == ()
        assert query.values == ()

    def test_embed_texts_batch(self) -> None:
        embedder = SparseEmbedder()
        results = embedder.embed_texts(["a b", "b c", ""])

        assert len(results) == 3
        assert len(results[0].indices) == 2
        assert len(results[1].indices) == 2
        assert len(results[2].indices) == 0

    def test_to_qdrant_format(self) -> None:
        embedder = SparseEmbedder()
        result = embedder.embed_text("test word")
        qdrant_vec = result.to_qdrant()

        assert qdrant_vec.indices == list(result.indices)
        assert qdrant_vec.values == list(result.values)

    def test_clear_resets_vocab(self) -> None:
        embedder = SparseEmbedder()
        embedder.embed_text("hello")
        assert embedder.vocab_size == 1

        embedder.clear()
        assert embedder.vocab_size == 0

    def test_case_insensitive(self) -> None:
        embedder = SparseEmbedder()
        r1 = embedder.embed_text("Hello World")
        r2 = embedder.embed_text("hello world")

        assert r1.indices == r2.indices
        assert r1.values == r2.values

    def test_punctuation_stripped(self) -> None:
        embedder = SparseEmbedder()
        embedder.embed_text("hello, world! test.")

        assert embedder.vocab_size == 3
