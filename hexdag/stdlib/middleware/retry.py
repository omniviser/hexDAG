"""Retry middleware with exponential backoff.

Wraps any port and retries failed calls with configurable exponential backoff.
Designed for transient failures (HTTP 429, 503, network errors).

Example YAML::

    spec:
      ports:
        llm:
          adapter: hexdag.stdlib.adapters.openai.OpenAIAdapter
          middleware:
            - hexdag.stdlib.middleware.retry.RetryWithBackoff
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

from hexdag.kernel.logging import get_logger

logger = get_logger(__name__)

# Default retry configuration
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BASE_DELAY = 1.0
_DEFAULT_MAX_DELAY = 60.0
_DEFAULT_JITTER = True


class RetryWithBackoff:
    """Middleware that retries failed port calls with exponential backoff.

    Intercepts all async method calls on the inner port.  On failure,
    retries up to ``max_retries`` times with exponential backoff and
    optional jitter.

    Parameters
    ----------
    inner : Any
        The port/adapter to wrap.
    max_retries : int
        Maximum number of retry attempts (default 3).
    base_delay : float
        Initial delay in seconds before first retry (default 1.0).
    max_delay : float
        Maximum delay cap in seconds (default 60.0).
    jitter : bool
        Add random jitter to prevent thundering herd (default True).
    retryable_exceptions : tuple[type[Exception], ...] | None
        Exception types to retry on.  If None, retries on all exceptions.
    """

    def __init__(
        self,
        inner: Any,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        base_delay: float = _DEFAULT_BASE_DELAY,
        max_delay: float = _DEFAULT_MAX_DELAY,
        jitter: bool = _DEFAULT_JITTER,
        retryable_exceptions: tuple[type[Exception], ...] | None = None,
    ) -> None:
        """Initialize retry middleware with backoff configuration."""
        self._inner = inner
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._jitter = jitter
        self._retryable_exceptions = retryable_exceptions
        self._total_retries = 0

    @property
    def total_retries(self) -> int:
        """Total number of retries performed across all calls."""
        return self._total_retries

    def _compute_delay(self, attempt: int) -> float:
        """Compute delay for a given attempt number (0-indexed)."""
        delay: float = min(self._base_delay * (2**attempt), self._max_delay)
        if self._jitter:
            delay = delay * (0.5 + random.random() * 0.5)  # noqa: S311
        return delay

    def _is_retryable(self, exc: Exception) -> bool:
        """Check whether an exception should trigger a retry."""
        if self._retryable_exceptions is None:
            return True
        return isinstance(exc, self._retryable_exceptions)

    async def _call_with_retry(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Call a method on the inner port with retry logic."""
        method = getattr(self._inner, method_name)
        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                return await method(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt >= self._max_retries or not self._is_retryable(exc):
                    raise

                delay = self._compute_delay(attempt)
                self._total_retries += 1
                logger.warning(
                    "Retry {}/{} for {}.{} after {:.2f}s — {}",
                    attempt + 1,
                    self._max_retries,
                    type(self._inner).__name__,
                    method_name,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)

        raise last_exc  # type: ignore[misc]  # pragma: no cover

    # -- LLM protocol methods --

    async def aresponse(self, *args: Any, **kwargs: Any) -> Any:
        """Retry-wrapped aresponse."""
        return await self._call_with_retry("aresponse", *args, **kwargs)

    async def aresponse_with_tools(self, *args: Any, **kwargs: Any) -> Any:
        """Retry-wrapped aresponse_with_tools."""
        return await self._call_with_retry("aresponse_with_tools", *args, **kwargs)

    async def aresponse_structured(self, *args: Any, **kwargs: Any) -> Any:
        """Retry-wrapped aresponse_structured."""
        return await self._call_with_retry("aresponse_structured", *args, **kwargs)

    # -- ToolRouter protocol methods --

    async def acall_tool(self, *args: Any, **kwargs: Any) -> Any:
        """Retry-wrapped acall_tool."""
        return await self._call_with_retry("acall_tool", *args, **kwargs)

    # -- Passthrough --

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the inner port."""
        return getattr(self._inner, name)
