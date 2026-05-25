from __future__ import annotations

from rag.generation.prompt_builder import SYSTEM_PROMPT, build_rag_prompt
from rag.models.document import Chunk, ChunkMetadata
from rag.models.retrieval import ScoredChunk


def _make_scored_chunk(
    text: str, source: str = "doc.pdf", score: float = 0.9
) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            document_id="doc-1",
            text=text,
            metadata=ChunkMetadata(source_file=source),
        ),
        score=score,
        retrieval_method="vector",
    )


class TestBuildRagPrompt:
    def test_formats_three_chunks(self) -> None:
        chunks = [
            _make_scored_chunk("Alpha text", "alpha.pdf"),
            _make_scored_chunk("Bravo text", "bravo.txt"),
            _make_scored_chunk("Charlie text", "charlie.md"),
        ]
        system, user = build_rag_prompt("What is alpha?", chunks)

        assert system == SYSTEM_PROMPT
        assert "[1] (source: alpha.pdf) Alpha text" in user
        assert "[2] (source: bravo.txt) Bravo text" in user
        assert "[3] (source: charlie.md) Charlie text" in user
        assert "Question: What is alpha?" in user

    def test_empty_chunks_shows_no_sources(self) -> None:
        system, user = build_rag_prompt("What is this?", [])

        assert system == SYSTEM_PROMPT
        assert "No sources available" in user
        assert "Question: What is this?" in user

    def test_system_prompt_contains_grounding(self) -> None:
        assert "ONLY using information from the provided sources" in SYSTEM_PROMPT

    def test_system_prompt_contains_citation_format(self) -> None:
        assert "[1], [2]" in SYSTEM_PROMPT

    def test_system_prompt_contains_abstention(self) -> None:
        assert "I don't have enough information" in SYSTEM_PROMPT

    def test_system_prompt_contains_no_speculation(self) -> None:
        assert "Do NOT guess or speculate" in SYSTEM_PROMPT

    def test_system_prompt_contains_conflict_handling(self) -> None:
        assert "conflicting information" in SYSTEM_PROMPT

    def test_question_appears_after_sources(self) -> None:
        chunks = [_make_scored_chunk("Some text")]
        _, user = build_rag_prompt("My question?", chunks)
        sources_pos = user.index("Sources:")
        question_pos = user.index("Question:")
        assert question_pos > sources_pos

    def test_source_filenames_in_prompt(self) -> None:
        chunks = [_make_scored_chunk("text", "report.pdf")]
        _, user = build_rag_prompt("question", chunks)
        assert "report.pdf" in user

    def test_single_chunk(self) -> None:
        chunks = [_make_scored_chunk("Only one source")]
        _, user = build_rag_prompt("question", chunks)
        assert "[1]" in user
        assert "[2]" not in user
