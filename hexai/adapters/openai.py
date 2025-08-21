"""OpenAI adapter for hexai framework."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Awaitable
from typing import Any, Callable, Literal, Optional

try:
    from hexai.helpers.secrets import get_secret as _get_secret
except Exception:  # fallback

    # TODO: replace with hexai.helpers.secrets.get_secret
    # once Wojtek's secret handler is merged into main

    def _get_secret(key: str, default: Optional[str] = None, required: bool = False) -> str:
        val = os.getenv(key, default)
        if required and val is None:
            raise RuntimeError(f"Missing required secret: {key}")
        return val or ""


SupportedProvider = Literal["openai", "azure", "anthropic", "gemini", "ollama"]


class TokenBucket:
    """Asynchronous token-bucket rate limiter.

    Parameters
    ----------
    rate_per_s : float
        Number of tokens refilled per second.
    capacity : int
        Maximum number of tokens that can be stored in the bucket.
    """

    def __init__(self, rate_per_s: float, capacity: int) -> None:
        self.rate = rate_per_s
        self.capacity = capacity
        self.tokens = float(capacity)
        self.updated = asyncio.get_event_loop().time()
        self._lock = asyncio.Lock()

    async def acquire(self, cost: int = 1) -> None:
        """Acquire a given number of tokens, waiting if necessary.

        Parameters
        ----------
        cost : int, optional
            Number of tokens required for the operation (default is 1).

        Raises
        ------
        asyncio.CancelledError
            If the waiting coroutine is cancelled before tokens are available.
        """
        async with self._lock:
            now = asyncio.get_event_loop().time()
            # Refill tokens based on elapsed time
            self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
            self.updated = now

            while self.tokens < cost:
                needed = cost - self.tokens
                await asyncio.sleep(max(0.0, needed / self.rate))
                now = asyncio.get_event_loop().time()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
                self.updated = now

            self.tokens -= cost


class OpenAIAdapter:
    """OpenAI-compatible adapter (skeleton)."""

    def __init__(
        self,
        *,
        provider: SupportedProvider = "openai",
        model: Optional[str] = None,
        secrets_provider: Callable[[str, Optional[str]], str] = _get_secret,
        **client_kwargs: Any,
    ) -> None:
        self.provider = provider
        self.model = model
        self._get_secret = secrets_provider
        self._client_kwargs = client_kwargs
        self._client: Any | None = None  # explicit type for mypy

    def _require_secret(self, key: str) -> str:
        """Return a non-empty secret value or raise a clear error.

        Parameters
        ----------
        key : str
            Environment/config variable name to fetch.

        Returns
        -------
        str
            Non-empty secret value.

        Raises
        ------
        RuntimeError
            If the secret is missing or empty.
        """
        val = self._get_secret(key, None)
        if not val:
            raise RuntimeError(f"Missing required secret: {key}")
        return val

    async def _ensure_client(self) -> None:
        """Lazily initialize the underlying SDK client for the selected provider.

        Notes
        -----
        MVP supports ``openai`` and ``azure`` using the OpenAI Python SDK (async).
        Other providers will be added in subsequent steps.
        """
        # if already initialized -> exit early
        if self._client is not None:
            return None

        # only these providers supported in MVP
        if self.provider not in ("openai", "azure"):
            raise NotImplementedError(f"Provider '{self.provider}' not implemented yet.")

        # lazy import so unit tests don't need the SDK installed
        try:
            from openai import AsyncOpenAI
        except Exception as e:  # pragma: no cover
            raise RuntimeError("OpenAI SDK not installed. Add `openai` to dependencies.") from e

        api_key = self._require_secret(
            "AZURE_OPENAI_API_KEY" if self.provider == "azure" else "OPENAI_API_KEY"
        )
        base_url = (
            os.getenv("AZURE_OPENAI_ENDPOINT")
            if self.provider == "azure"
            else os.getenv("OPENAI_BASE_URL")
        )

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, **self._client_kwargs)
        return None

    async def astream(
        self,
        *,
        messages: list[dict[str, Any]],
        model: Optional[str] = None,
        **params: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream chat completions as an async iterator (MVP for OpenAI/Azure).

        Parameters
        ----------
        messages : list of dict
            Chat messages in OpenAI format, e.g. [{"role": "user", "content": "..."}].
        model : str, optional
            Model override for this call; falls back to self.model if not provided.
        **params : Any
            Extra parameters forwarded to the provider SDK.

        Yields
        ------
        dict
            Normalized stream deltas, e.g. {"type": "content", "data": "<chunk>"} or
            {"type": "finish", "data": "<finish_reason>"}.
        """
        bucket = TokenBucket(rate_per_s=5, capacity=10)
        await bucket.acquire(1)

        await self._ensure_client()
        if self._client is None:
            raise RuntimeError("SDK client not initialized")
        client = self._client

        async def _call_stream() -> Any:
            return await client.chat.completions.create(
                model=model or self.model,
                messages=list(messages),
                stream=True,
                **params,
            )

        stream = await self._with_retries(_call_stream)

        async for chunk in stream:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            choice = choices[0]

            delta = getattr(choice, "delta", None)
            if delta is not None and getattr(delta, "content", None):
                yield {"type": "content", "data": delta.content}

            finish = getattr(choice, "finish_reason", None)
            if finish:
                yield {"type": "finish", "data": finish}

    async def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        model: Optional[str] = None,
        **params: Any,
    ) -> dict[str, Any]:
        """Non-streaming chat completion call.

        Parameters
        ----------
        messages : list of dict
            OpenAI-format chat messages, e.g. [{"role": "user", "content": "hi"}].
        model : str, optional
            Model override for this call; falls back to self.model if not provided.
        **params : Any
            Extra provider parameters passed through to the SDK.

        Returns
        -------
        dict
            Dictionary with keys:
            - "content": str | None
            - "usage": Any
            - "raw": Any
        """
        # simple per-call rate limit (MVP)
        bucket = TokenBucket(rate_per_s=5, capacity=10)
        await bucket.acquire(1)

        await self._ensure_client()
        if self._client is None:
            raise RuntimeError("SDK client not initialized")
        client = self._client  # <- zawężamy typ

        async def _call() -> Any:
            return await client.chat.completions.create(
                model=model or self.model,
                messages=list(messages),
                stream=False,
                **params,
            )

        resp = await self._with_retries(_call)

        content = None
        if getattr(resp, "choices", None):
            msg = getattr(resp.choices[0], "message", None)
            content = getattr(msg, "content", None) if msg else None

        usage = getattr(resp, "usage", None)
        return {"content": content, "usage": usage, "raw": resp}

    async def _with_retries(
        self,
        op: Callable[[], Awaitable[Any]],
        *,
        max_attempts: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 4.0,
    ) -> Any:
        """Run async operation with exponential backoff + jitter on transient errors.

        Retries:
        - timeouts
        - HTTP 429 / 5xx (if exception exposes `.status` or `.status_code`)
        - textual hints in error message ("rate limit", "temporarily unavailable")

        Parameters
        ----------
        op : callable
            Zero-arg async function that performs the SDK call.
        max_attempts : int
            Maximum number of attempts (including the first one).
        base_delay : float
            Initial backoff in seconds.
        max_delay : float
            Upper bound for backoff.

        Returns
        -------
        Any
            Result of `op()` if it eventually succeeds.

        Raises
        ------
        Exception
            Last error if all attempts fail or error is non-retryable.
        """
        attempt = 0
        while True:
            try:
                return await op()
            except Exception as e:  # noqa: BLE001 - we normalize retryability below
                attempt += 1

                # Decide if retryable
                status = getattr(e, "status", None) or getattr(e, "status_code", None)
                msg = str(e).lower()
                retryable = (
                    isinstance(e, asyncio.TimeoutError)
                    or status == 429
                    or (isinstance(status, int) and 500 <= status < 600)
                    or "rate limit" in msg
                    or "temporarily unavailable" in msg
                    or "timeout" in msg
                )

                if not retryable or attempt >= max_attempts:
                    raise RuntimeError(f"Request failed after {attempt} attempt(s): {e}") from e

                # exponential backoff + jitter
                delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                jitter = delay * 0.3
                sleep_s = max(0.0, delay - jitter) + (
                    jitter * (os.getpid() % 10) / 10.0
                )  # deterministic-ish
                await asyncio.sleep(sleep_s)
