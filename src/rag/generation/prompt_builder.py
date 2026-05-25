from __future__ import annotations

from collections.abc import Sequence

from rag.models.retrieval import ScoredChunk

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based strictly "
    "on the provided source documents.\n\n"
    "Rules:\n"
    "1. Answer ONLY using information from the provided sources.\n"
    "2. Cite sources inline using [1], [2], etc. after each factual claim.\n"
    "3. If multiple sources support a claim, cite all of them: [1][3].\n"
    "4. If the sources do not contain enough information to answer the "
    "question, say: \"I don't have enough information in the available "
    'sources to answer this question."\n'
    "5. Do NOT use your general knowledge. Do NOT guess or speculate.\n"
    "6. If sources contain conflicting information, present both "
    "perspectives with their citations."
)


def build_rag_prompt(
    question: str,
    scored_chunks: Sequence[ScoredChunk],
) -> tuple[str, str]:
    if not scored_chunks:
        user_prompt = f"Sources:\nNo sources available.\n\nQuestion: {question}"
        return SYSTEM_PROMPT, user_prompt

    source_lines = []
    for i, sc in enumerate(scored_chunks, start=1):
        source_file = sc.chunk.metadata.source_file
        text = sc.chunk.text
        source_lines.append(f"[{i}] (source: {source_file}) {text}")

    sources_block = "\n\n".join(source_lines)
    user_prompt = f"Sources:\n{sources_block}\n\nQuestion: {question}"

    return SYSTEM_PROMPT, user_prompt
