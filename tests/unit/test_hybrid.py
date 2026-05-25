from __future__ import annotations

from rag.models.document import Chunk, ChunkMetadata
from rag.models.retrieval import ScoredChunk
from rag.retrieval.hybrid import reciprocal_rank_fusion


def _make_sc(chunk_id: str, score: float, method: str = "vector") -> ScoredChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        text=f"text for {chunk_id}",
        metadata=ChunkMetadata(source_file="test.txt"),
    )
    return ScoredChunk(chunk=chunk, score=score, retrieval_method=method)


class TestReciprocalRankFusion:
    def test_single_list(self) -> None:
        results = [_make_sc("c1", 0.9), _make_sc("c2", 0.8)]
        fused = reciprocal_rank_fusion(results)

        assert len(fused) == 2
        assert fused[0].chunk.chunk_id == "c1"
        assert fused[1].chunk.chunk_id == "c2"

    def test_two_lists_merge(self) -> None:
        vector = [_make_sc("c1", 0.9), _make_sc("c2", 0.8)]
        bm25 = [_make_sc("c2", 5.0, "bm25"), _make_sc("c3", 4.0, "bm25")]

        fused = reciprocal_rank_fusion(vector, bm25)

        assert len(fused) == 3
        # c2 appears in both lists, should have highest RRF score
        assert fused[0].chunk.chunk_id == "c2"

    def test_all_results_have_hybrid_method(self) -> None:
        vector = [_make_sc("c1", 0.9)]
        bm25 = [_make_sc("c2", 5.0, "bm25")]

        fused = reciprocal_rank_fusion(vector, bm25)

        for sc in fused:
            assert sc.retrieval_method == "hybrid"

    def test_scores_are_rrf_values(self) -> None:
        vector = [_make_sc("c1", 0.9)]
        fused = reciprocal_rank_fusion(vector, k=60)

        # RRF score = 1 / (60 + 1) for rank 1 in one list
        expected = 1.0 / 61
        assert abs(fused[0].score - expected) < 1e-9

    def test_document_in_both_lists_gets_higher_score(self) -> None:
        vector = [_make_sc("c1", 0.9), _make_sc("c2", 0.8)]
        bm25 = [_make_sc("c1", 5.0, "bm25")]

        fused = reciprocal_rank_fusion(vector, bm25)

        scores = {sc.chunk.chunk_id: sc.score for sc in fused}
        # c1 in both lists, c2 in only one
        assert scores["c1"] > scores["c2"]

    def test_empty_lists(self) -> None:
        fused = reciprocal_rank_fusion([], [])
        assert fused == ()

    def test_one_empty_one_populated(self) -> None:
        vector = [_make_sc("c1", 0.9)]
        fused = reciprocal_rank_fusion(vector, [])

        assert len(fused) == 1
        assert fused[0].chunk.chunk_id == "c1"

    def test_preserves_chunk_data(self) -> None:
        vector = [_make_sc("c1", 0.9)]
        fused = reciprocal_rank_fusion(vector)

        assert fused[0].chunk.text == "text for c1"
        assert fused[0].chunk.document_id == "doc-1"

    def test_three_lists(self) -> None:
        list1 = [_make_sc("c1", 0.9)]
        list2 = [_make_sc("c1", 5.0, "bm25")]
        list3 = [_make_sc("c1", 0.7)]

        fused = reciprocal_rank_fusion(list1, list2, list3, k=60)

        # c1 in all three lists at rank 1: 3 * 1/(60+1)
        expected = 3.0 / 61
        assert abs(fused[0].score - expected) < 1e-9
