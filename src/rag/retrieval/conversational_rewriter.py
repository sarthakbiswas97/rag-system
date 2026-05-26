from __future__ import annotations

import logging

from rag.generation.llm_client import LLMClient
from rag.models.session import Session

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a query rewriter for a retrieval-augmented generation system. "
    "Given a conversation history and the user's latest question, rewrite the "
    "question as a standalone query that captures the full context.\n\n"
    "Rules:\n"
    "- If the question is already standalone, return it unchanged.\n"
    "- Resolve all pronouns and references using the conversation history.\n"
    "- Keep the rewritten query concise and search-friendly.\n"
    "- Output ONLY the rewritten query, nothing else."
)

MIN_HISTORY_TURNS = 1


def _build_user_prompt(session: Session, current_query: str) -> str:
    history_lines = [
        f"{t.role}: {t.content}"
        for t in session.turns[:-1]  # exclude the current turn
    ]
    history_block = "\n".join(history_lines)
    return (
        f"Conversation history:\n{history_block}\n\n"
        f"Latest question: {current_query}\n\n"
        f"Rewritten standalone query:"
    )


async def rewrite_with_context(
    query: str,
    session: Session | None,
    llm_client: LLMClient,
) -> str:
    """Rewrite a query using conversation history for context.

    Returns the original query if no session or insufficient history.
    """
    if session is None or len(session.turns) < MIN_HISTORY_TURNS + 1:
        return query

    user_prompt = _build_user_prompt(session, query)

    try:
        response = await llm_client.generate(_SYSTEM_PROMPT, user_prompt)
        rewritten = response.content.strip()

        if not rewritten:
            return query

        logger.info(
            "Conversational rewrite complete",
            extra={
                "original": query[:100],
                "rewritten": rewritten[:100],
                "history_turns": len(session.turns) - 1,
            },
        )
        return rewritten

    except Exception:
        logger.warning(
            "Conversational rewrite failed, using original query",
            exc_info=True,
        )
        return query
