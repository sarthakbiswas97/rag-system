from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import APIConnectionError

from rag.generation.llm_client import LLMClient, LLMServiceError


def _make_connection_error(msg: str = "failed") -> APIConnectionError:
    request = MagicMock()
    return APIConnectionError(message=msg, request=request)


def _mock_response(
    content: str = "Test response",
    model: str = "gpt-4o-mini",
) -> MagicMock:
    usage = MagicMock()
    usage.prompt_tokens = 50
    usage.completion_tokens = 20

    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    response.model = model
    return response


class TestLLMClientFallback:
    @pytest.mark.asyncio
    async def test_fallback_called_on_primary_failure(self) -> None:
        LLMClient(api_key="primary-key")
        fallback = LLMClient(api_key="fallback-key")

        primary_with_fallback = LLMClient(
            api_key="primary-key",
            fallback_client=fallback,
        )

        # Mock primary to fail, fallback to succeed
        with patch.object(
            primary_with_fallback._client.chat.completions,
            "create",
            new=AsyncMock(side_effect=_make_connection_error("primary failed")),
        ), patch.object(
            fallback._client.chat.completions,
            "create",
            new=AsyncMock(return_value=_mock_response("fallback answer")),
        ):
            result = await primary_with_fallback.generate("sys", "usr")

        assert result.content == "fallback answer"

    @pytest.mark.asyncio
    async def test_fallback_not_called_when_primary_succeeds(self) -> None:
        fallback = MagicMock(spec=LLMClient)
        fallback.generate = AsyncMock(return_value=_mock_response("fallback"))

        primary = LLMClient(
            api_key="primary-key",
            fallback_client=fallback,
        )

        with patch.object(
            primary._client.chat.completions,
            "create",
            new=AsyncMock(return_value=_mock_response("primary answer")),
        ):
            result = await primary.generate("sys", "usr")

        assert result.content == "primary answer"
        fallback.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_when_no_fallback_and_fails(self) -> None:
        primary = LLMClient(api_key="primary-key")

        with patch.object(
            primary._client.chat.completions,
            "create",
            new=AsyncMock(side_effect=_make_connection_error("service down")),
        ), pytest.raises(LLMServiceError):
            await primary.generate("sys", "usr")

    @pytest.mark.asyncio
    async def test_fallback_also_fails_raises_original(self) -> None:
        fallback = LLMClient(api_key="fallback-key")

        primary = LLMClient(
            api_key="primary-key",
            fallback_client=fallback,
        )

        with patch.object(
            primary._client.chat.completions,
            "create",
            new=AsyncMock(side_effect=_make_connection_error("primary failed")),
        ), patch.object(
            fallback._client.chat.completions,
            "create",
            new=AsyncMock(side_effect=_make_connection_error("fallback failed")),
        ), pytest.raises(LLMServiceError):
            await primary.generate("sys", "usr")

    def test_fallback_client_stored(self) -> None:
        fallback = LLMClient(api_key="fallback-key")
        primary = LLMClient(
            api_key="primary-key",
            fallback_client=fallback,
        )
        assert primary._fallback_client is fallback
