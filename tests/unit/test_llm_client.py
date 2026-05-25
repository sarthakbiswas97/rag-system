from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag.generation.llm_client import LLMClient


def _mock_response(content: str = "Test response") -> MagicMock:
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
    return response


class TestLLMClientInit:
    def test_raises_on_empty_api_key(self) -> None:
        with pytest.raises(ValueError, match="API key is required"):
            LLMClient(api_key="")

    def test_creates_with_valid_key(self) -> None:
        client = LLMClient(api_key="test-key")
        assert client._model == "gpt-4o-mini"

    def test_accepts_custom_params(self) -> None:
        client = LLMClient(
            api_key="test-key",
            model="gpt-4o",
            temperature=0.0,
            max_tokens=2048,
            timeout=60.0,
        )
        assert client._model == "gpt-4o"
        assert client._temperature == 0.0
        assert client._max_tokens == 2048


class TestLLMClientGenerate:
    @pytest.fixture()
    def client(self) -> LLMClient:
        return LLMClient(api_key="test-key")

    @pytest.mark.asyncio
    async def test_returns_llm_response(self, client: LLMClient) -> None:
        mock_resp = _mock_response("Paris is the capital [1].")
        with patch.object(
            client._client.chat.completions,
            "create",
            new=AsyncMock(return_value=mock_resp),
        ):
            result = await client.generate("system prompt", "user prompt")

        assert result.content == "Paris is the capital [1]."
        assert result.prompt_tokens == 50
        assert result.completion_tokens == 20
        assert result.model == "gpt-4o-mini"
        assert result.elapsed_ms > 0

    @pytest.mark.asyncio
    async def test_passes_correct_messages(self, client: LLMClient) -> None:
        mock_create = AsyncMock(return_value=_mock_response())
        with patch.object(client._client.chat.completions, "create", new=mock_create):
            await client.generate("sys", "usr")

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "sys"}
        assert messages[1] == {"role": "user", "content": "usr"}

    @pytest.mark.asyncio
    async def test_passes_model_and_temperature(self, client: LLMClient) -> None:
        mock_create = AsyncMock(return_value=_mock_response())
        with patch.object(client._client.chat.completions, "create", new=mock_create):
            await client.generate("sys", "usr")

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["temperature"] == 0.1
        assert call_kwargs["max_tokens"] == 1024

    @pytest.mark.asyncio
    async def test_handles_none_content(self, client: LLMClient) -> None:
        mock_resp = _mock_response()
        mock_resp.choices[0].message.content = None
        with patch.object(
            client._client.chat.completions,
            "create",
            new=AsyncMock(return_value=mock_resp),
        ):
            result = await client.generate("sys", "usr")

        assert result.content == ""

    @pytest.mark.asyncio
    async def test_handles_none_usage(self, client: LLMClient) -> None:
        mock_resp = _mock_response()
        mock_resp.usage = None
        with patch.object(
            client._client.chat.completions,
            "create",
            new=AsyncMock(return_value=mock_resp),
        ):
            result = await client.generate("sys", "usr")

        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
