"""Unified LLM adapter using OpenAI-compatible API.

All 5 Chinese LLM providers support the OpenAI-compatible /v1/chat/completions endpoint.
Uses httpx for lightweight async HTTP calls without the openai SDK dependency.
"""

import asyncio
import json
import time
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

        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")

        self._client: httpx.AsyncClient | None = None
        self._oauth_token: str | None = None
        self._oauth_token_expiry: float = 0.0

    async def _get_oauth_token(self) -> str:
        """Exchange API key for OAuth access token (Baidu ERNIE)."""
        if self._oauth_token and time.time() < self._oauth_token_expiry - 60:
            return self._oauth_token

        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": self.provider.api_key,
            "client_secret": self.provider.api_key,
        }
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            resp = await client.post(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            token = data.get("access_token")
            if not token:
                error_keys = {k for k in data if k not in ("access_token", "expires_in", "token_type")}
                raise RuntimeError(
                    f"OAuth token endpoint returned no access_token. "
                    f"Response keys: {sorted(error_keys) if error_keys else 'none'}"
                )
            self._oauth_token = token
            self._oauth_token_expiry = time.time() + data.get("expires_in", 86400)
            return self._oauth_token

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            base_url = self.provider.base_url
            if not base_url.endswith("/"):
                base_url += "/"
            self._client = httpx.AsyncClient(
                base_url=base_url,
                headers={
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(60.0),
            )
        return self._client

    async def _get_auth_headers(self) -> dict[str, str]:
        """Get authorization headers based on provider auth type."""
        if self.provider.auth_type == "oauth":
            token = await self._get_oauth_token()
            return {"Authorization": f"Bearer {token}"}
        return {"Authorization": f"Bearer {self.provider.api_key}"}

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
                auth_headers = await self._get_auth_headers()
                headers = {**auth_headers, "Content-Type": "application/json"}
                response = await self.client.post(
                    "chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                return response
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                # HTTPStatusError is NOT a subclass of RequestError in httpx,
                # so both must be caught explicitly.
                if isinstance(e, httpx.HTTPStatusError):
                    status = e.response.status_code
                    # Always close the response body before retrying/raising
                    await e.response.aclose()
                    # Don't retry on permanent client errors (4xx except 429)
                    if status < 500 and status != 429:
                        raise
                if attempt < self.max_retries - 1:
                    wait = self.backoff_seconds * (2 ** attempt)
                    await asyncio.sleep(wait)
                else:
                    raise RuntimeError(
                        f"LLM request failed after {self.max_retries} attempts"
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

        try:
            async for line in response.aiter_lines():
                # SSE spec: "data:" followed by an optional space. Accept both
                # "data: {...}" and "data:{...}" so we don't drop tokens from
                # providers that omit the space.
                if line.startswith("data:"):
                    data = line[5:].lstrip()
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
        finally:
            await response.aclose()

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
        finally:
            await response.aclose()

    async def close(self):
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None