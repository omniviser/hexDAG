"""OpenAI adapter for hexai framework."""

from __future__ import annotations

import asyncio
import os
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
        # no code after this point that could be flagged as unreachable
        return None
