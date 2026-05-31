from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from rag.models.generation import LLMResponse
from rag.models.session import Session
from rag.retrieval.conversational_rewriter import (
    _build_user_prompt,
    rewrite_with_context,
)


def _make_llm_client(response_text: str) -> MagicMock:
    client = MagicMock()
    client.generate = AsyncMock(
        return_value=LLMResponse(
            content=response_text,
            prompt_tokens=10,
            completion_tokens=5,
            model="test",
            elapsed_ms=50.0,
        )
    )
    return client


def _make_session_with_history() -> Session:
    session = Session(tenant_id="t1")
    session = session.add_turn("user", "What is the capital of France?")
    session = session.add_turn("assistant", "The capital of France is Paris.")
    session = session.add_turn("user", "What about its population?")
    return session


class TestRewriteWithContext:
    @pytest.mark.asyncio
    async def test_rewrites_with_history(self) -> None:
        client = _make_llm_client("What is the population of Paris?")
        session = _make_session_with_history()

        result = await rewrite_with_context(
            "What about its population?", session, client
        )

        assert result == "What is the population of Paris?"
        client.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_original_when_no_session(self) -> None:
        client = _make_llm_client("rewritten")

        result = await rewrite_with_context("standalone query", None, client)

        assert result == "standalone query"
        client.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_original_when_insufficient_history(self) -> None:
        client = _make_llm_client("rewritten")
        session = Session()
        session = session.add_turn("user", "First question")

        result = await rewrite_with_context("First question", session, client)

        assert result == "First question"
        client.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_original_on_llm_failure(self) -> None:
        client = MagicMock()
        client.generate = AsyncMock(side_effect=RuntimeError("API down"))
        session = _make_session_with_history()

        result = await rewrite_with_context(
            "What about its population?", session, client
        )

        assert result == "What about its population?"

    @pytest.mark.asyncio
    async def test_returns_original_on_empty_response(self) -> None:
        client = _make_llm_client("")
        session = _make_session_with_history()

        result = await rewrite_with_context(
            "What about its population?", session, client
        )

        assert result == "What about its population?"

    @pytest.mark.asyncio
    async def test_strips_whitespace_from_response(self) -> None:
        client = _make_llm_client("  rewritten query  \n")
        session = _make_session_with_history()

        result = await rewrite_with_context("query", session, client)

        assert result == "rewritten query"

    @pytest.mark.asyncio
    async def test_two_turn_history_triggers_rewrite(self) -> None:
        client = _make_llm_client("rewritten")
        session = Session()
        session = session.add_turn("user", "Tell me about France")
        session = session.add_turn("assistant", "France is a country")
        session = session.add_turn("user", "What about it?")

        result = await rewrite_with_context("What about it?", session, client)

        assert result == "rewritten"
        client.generate.assert_called_once()


class TestBuildUserPrompt:
    def test_includes_history_and_query(self) -> None:
        session = _make_session_with_history()

        prompt = _build_user_prompt(session, "What about its population?")

        assert "capital of France" in prompt
        assert "Paris" in prompt
        assert "What about its population?" in prompt
        assert "Rewritten standalone query:" in prompt

    def test_excludes_current_turn(self) -> None:
        session = _make_session_with_history()

        prompt = _build_user_prompt(session, "What about its population?")

        lines = prompt.split("\n")
        history_section = [
            line
            for line in lines
            if line.startswith("user:") or line.startswith("assistant:")
        ]
        # Should have 2 history turns, not 3 (current turn excluded)
        assert len(history_section) == 2
