from __future__ import annotations

import logging
import time

from openai import APIConnectionError, AsyncOpenAI, AuthenticationError, RateLimitError

from rag.models.generation import LLMResponse

logger = logging.getLogger(__name__)


class LLMServiceError(Exception):
    """Raised when the LLM provider returns an error."""


class LLMClient:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: float = 30.0,
        fallback_client: LLMClient | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OpenAI API key is required")

        self._client = AsyncOpenAI(
            api_key=api_key,
            max_retries=3,
            timeout=timeout,
        )
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._fallback_client = fallback_client

        logger.info(
            "LLMClient initialized",
            extra={
                "model": model,
                "max_tokens": max_tokens,
                "has_fallback": fallback_client is not None,
            },
        )

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        try:
            return await self._generate_internal(system_prompt, user_prompt)
        except LLMServiceError as exc:
            if self._fallback_client is not None:
                logger.warning(
                    "Primary LLM failed, trying fallback",
                    extra={
                        "primary_model": self._model,
                        "error": str(exc),
                    },
                )
                return await self._fallback_client.generate(system_prompt, user_prompt)
            raise

    async def stream_generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ):
        """Yield tokens and return final accumulated response.

        .. note::
            Fallback is not supported for streaming. If the primary model
            fails, the error is raised directly to the caller.
        """
        start = time.perf_counter()

        try:
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                stream=True,
            )
        except AuthenticationError as exc:
            raise LLMServiceError("LLM authentication failed — check API key") from exc
        except RateLimitError as exc:
            raise LLMServiceError("LLM rate limit exceeded — try again later") from exc
        except APIConnectionError as exc:
            raise LLMServiceError("LLM service unreachable") from exc

        content_parts: list[str] = []
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                content_parts.append(delta)
                yield delta

        elapsed_ms = (time.perf_counter() - start) * 1000
        full_content = "".join(content_parts)

        logger.info(
            "LLM streaming complete",
            extra={
                "model": self._model,
                "completion_tokens": len(content_parts),
                "elapsed_ms": round(elapsed_ms, 1),
            },
        )

        yield LLMResponse(
            content=full_content,
            prompt_tokens=0,  # Not available in streaming mode
            completion_tokens=len(content_parts),
            model=self._model,
            elapsed_ms=round(elapsed_ms, 1),
        )

    async def _generate_internal(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        start = time.perf_counter()

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except AuthenticationError as exc:
            raise LLMServiceError("LLM authentication failed — check API key") from exc
        except RateLimitError as exc:
            raise LLMServiceError("LLM rate limit exceeded — try again later") from exc
        except APIConnectionError as exc:
            raise LLMServiceError("LLM service unreachable") from exc

        elapsed_ms = (time.perf_counter() - start) * 1000

        content = response.choices[0].message.content or ""
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        logger.info(
            "LLM generation complete",
            extra={
                "model": self._model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "elapsed_ms": round(elapsed_ms, 1),
            },
        )

        return LLMResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=self._model,
            elapsed_ms=round(elapsed_ms, 1),
        )
