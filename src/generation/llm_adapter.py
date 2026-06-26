"""Unified LLM adapter using OpenAI-compatible API.

All 5 Chinese LLM providers support the OpenAI-compatible /v1/chat/completions endpoint.
Uses httpx for lightweight async HTTP calls without the openai SDK dependency.
"""

import asyncio
import json
from typing import Any, AsyncIterator

import httpx

from src.generation.providers import LLMProviderConfig


class LLMAdapter:
    """Unified LLM client for OpenAI-compatible chat completion APIs."""

    def __init__(
        self,
        provider: LLMProviderConfig,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
    ):
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.provider.base_url,
                headers={
                    "Authorization": f"Bearer {self.provider.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(60.0),
            )
        return self._client

    async def _chat_completion_request(
        self,
        messages: list[dict[str, str]],
        stream: bool = True,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        """Send a chat completion request to the LLM provider."""
        payload = {
            "model": self.provider.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "stream": stream,
        }

        for attempt in range(self.max_retries):
            try:
                response = await self.client.post(
                    "/v1/chat/completions",
                    json=payload,
                )
                response.raise_for_status()
                return response
            except (httpx.HTTPError, httpx.TimeoutException) as e:
                # Don't retry on permanent client errors (4xx except 429 rate limit)
                if isinstance(e, httpx.HTTPStatusError) and e.response.status_code < 500 and e.response.status_code != 429:
                    raise
                if attempt < self.max_retries - 1:
                    wait = self.backoff_seconds * (2 ** attempt)
                    await asyncio.sleep(wait)
                else:
                    raise RuntimeError(
                        f"LLM request failed after {self.max_retries} attempts: {e}"
                    ) from e

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream chat completion tokens.

        Yields content tokens as they arrive from the SSE stream.
        """
        response = await self._chat_completion_request(
            messages, stream=True, temperature=temperature, max_tokens=max_tokens
        )

        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                except json.JSONDecodeError:
                    continue

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Non-streaming chat completion. Returns the full response."""
        response = await self._chat_completion_request(
            messages, stream=False, temperature=temperature, max_tokens=max_tokens
        )
        try:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise RuntimeError(
                f"Unexpected LLM response format: {e}"
            ) from e

    async def close(self):
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None