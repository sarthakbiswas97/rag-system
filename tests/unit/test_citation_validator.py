from __future__ import annotations

from unittest.mock import MagicMock

from rag.models.document import Chunk, ChunkMetadata
from rag.models.generation import Citation
from rag.models.retrieval import ScoredChunk
from rag.verification.citation_validator import (
    CitationValidator,
    filter_supported_citations,
)


def _make_scored_chunk(chunk_id: str, text: str, score: float) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        text=text,
        metadata=ChunkMetadata(source_file="test.txt"),
    )
    return ScoredChunk(chunk=chunk, score=score, retrieval_method="vector")


def _make_citation(
    chunk_id: str, sentence_index: int, source: str = "test.txt"
) -> Citation:
    return Citation(
        chunk_id=chunk_id,
        source_file=source,
        text_snippet="snippet",
        sentence_index=sentence_index,
    )


def _make_validator(
    check_results: list[tuple[str, float]],
    support_threshold: float = 0.5,
) -> CitationValidator:
    """Create validator with mocked entailment checker."""
    mock_checker = MagicMock()
    call_count = 0

    def mock_check(premise: str, hypothesis: str) -> tuple[str, float]:
        nonlocal call_count
        if call_count < len(check_results):
            result = check_results[call_count]
        else:
            result = check_results[-1]
        call_count += 1
        return result

    mock_checker.check_sentence = mock_check
    return CitationValidator(
        entailment_checker=mock_checker,
        support_threshold=support_threshold,
    )


class TestCitationValidator:
    def test_supported_citation(self) -> None:
        validator = _make_validator([("entailment", 0.95)])
        chunks = [_make_scored_chunk("c1", "Paris is the capital of France.", 0.9)]
        citations = [_make_citation("c1", 0)]

        results = validator.validate(
            "Paris is the capital [1].",
            citations,
            chunks,
        )

        assert len(results) == 1
        assert results[0].is_supported is True
        assert results[0].entailment_score == 0.95

    def test_unsupported_citation(self) -> None:
        validator = _make_validator([("contradiction", 0.1)])
        chunks = [_make_scored_chunk("c1", "Berlin is in Germany.", 0.9)]
        citations = [_make_citation("c1", 0)]

        results = validator.validate(
            "Paris is the capital [1].",
            citations,
            chunks,
        )

        assert len(results) == 1
        assert results[0].is_supported is False
        assert results[0].entailment_score == 0.1

    def test_unknown_chunk_id(self) -> None:
        validator = _make_validator([("entailment", 0.95)])
        chunks = [_make_scored_chunk("c1", "Some text.", 0.9)]
        citations = [_make_citation("c_unknown", 0)]

        results = validator.validate(
            "A claim [1].",
            citations,
            chunks,
        )

        assert len(results) == 1
        assert results[0].is_supported is False
        assert results[0].entailment_score == 0.0

    def test_mixed_citations(self) -> None:
        validator = _make_validator([
            ("entailment", 0.9),
            ("contradiction", 0.1),
        ])
        chunks = [
            _make_scored_chunk("c1", "Paris is France's capital.", 0.9),
            _make_scored_chunk("c2", "Berlin is in Germany.", 0.8),
        ]
        citations = [
            _make_citation("c1", 0),
            _make_citation("c2", 1),
        ]

        results = validator.validate(
            "Paris is the capital [1]. Tokyo is in Japan [2].",
            citations,
            chunks,
        )

        assert len(results) == 2
        assert results[0].is_supported is True
        assert results[1].is_supported is False

    def test_empty_citations(self) -> None:
        validator = _make_validator([("entailment", 0.95)])
        chunks = [_make_scored_chunk("c1", "text", 0.9)]

        results = validator.validate("answer", [], chunks)

        assert results == ()

    def test_empty_chunks(self) -> None:
        validator = _make_validator([("entailment", 0.95)])
        citations = [_make_citation("c1", 0)]

        results = validator.validate("answer [1].", citations, [])

        assert results == ()

    def test_custom_support_threshold(self) -> None:
        validator = _make_validator(
            [("neutral", 0.6)],
            support_threshold=0.8,
        )
        chunks = [_make_scored_chunk("c1", "text", 0.9)]
        citations = [_make_citation("c1", 0)]

        results = validator.validate("A claim [1].", citations, chunks)

        assert results[0].is_supported is False  # 0.6 < 0.8

    def test_sentence_index_out_of_range(self) -> None:
        validator = _make_validator([("entailment", 0.95)])
        chunks = [_make_scored_chunk("c1", "text", 0.9)]
        citations = [_make_citation("c1", 99)]  # no sentence at index 99

        results = validator.validate("One sentence.", citations, chunks)

        assert results[0].is_supported is False
        assert results[0].entailment_score == 0.0


class TestFilterSupportedCitations:
    def test_filters_unsupported(self) -> None:
        validator = _make_validator([
            ("entailment", 0.9),
            ("contradiction", 0.1),
            ("entailment", 0.85),
        ])
        chunks = [
            _make_scored_chunk("c1", "text1", 0.9),
            _make_scored_chunk("c2", "text2", 0.8),
            _make_scored_chunk("c3", "text3", 0.7),
        ]
        citations = [
            _make_citation("c1", 0),
            _make_citation("c2", 1),
            _make_citation("c3", 2),
        ]

        verifications = validator.validate(
            "Claim one [1]. Claim two [2]. Claim three [3].",
            citations,
            chunks,
        )
        filtered = filter_supported_citations(verifications)

        assert len(filtered) == 2
        assert filtered[0].chunk_id == "c1"
        assert filtered[1].chunk_id == "c3"

    def test_empty_verifications(self) -> None:
        assert filter_supported_citations([]) == ()
