"""Token-bucket rate limiter middleware.

Wraps any port and limits the rate of calls using a token-bucket algorithm.
Requests that exceed the rate are delayed (not rejected), ensuring smooth
throughput without errors.

Example YAML::

    spec:
      ports:
        llm:
          adapter: hexdag.stdlib.adapters.openai.OpenAIAdapter
          middleware:
            - hexdag.stdlib.middleware.rate_limiter.RateLimiter
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from hexdag.kernel.logging import get_logger

logger = get_logger(__name__)

# Default configuration
_DEFAULT_MAX_CALLS = 10
_DEFAULT_PERIOD = 60.0


class RateLimiter:
    """Middleware that rate-limits port calls using a token-bucket algorithm.

    Calls exceeding the rate are delayed until a token is available.
    Thread-safe via asyncio.Lock.

    Parameters
    ----------
    inner : Any
        The port/adapter to wrap.
    max_calls : int
        Maximum number of calls allowed per period (default 10).
    period : float
        Time window in seconds for the rate limit (default 60.0).
    """

    def __init__(
        self,
        inner: Any,
        max_calls: int = _DEFAULT_MAX_CALLS,
        period: float = _DEFAULT_PERIOD,
    ) -> None:
        """Initialize rate limiter with token-bucket configuration."""
        self._inner = inner
        self._max_calls = max_calls
        self._period = period
        self._tokens = float(max_calls)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()
        self._total_waits = 0

    @property
    def total_waits(self) -> int:
        """Total number of times a call had to wait for a token."""
        return self._total_waits

    async def _acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            # Refill tokens based on elapsed time
            self._tokens = min(
                self._max_calls,
                self._tokens + elapsed * (self._max_calls / self._period),
            )
            self._last_refill = now

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return

            # Must wait for token replenishment
            wait_time = (1.0 - self._tokens) * (self._period / self._max_calls)
            self._total_waits += 1
            logger.debug("Rate limit: waiting {:.2f}s for token", wait_time)

        # Wait outside lock so other coroutines can proceed
        await asyncio.sleep(wait_time)

        # Re-acquire after waiting
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(
                self._max_calls,
                self._tokens + elapsed * (self._max_calls / self._period),
            )
            self._last_refill = now
            self._tokens -= 1.0

    # -- LLM protocol methods --

    async def aresponse(self, *args: Any, **kwargs: Any) -> Any:
        """Rate-limited aresponse."""
        await self._acquire()
        return await self._inner.aresponse(*args, **kwargs)

    async def aresponse_with_tools(self, *args: Any, **kwargs: Any) -> Any:
        """Rate-limited aresponse_with_tools."""
        await self._acquire()
        return await self._inner.aresponse_with_tools(*args, **kwargs)

    async def aresponse_structured(self, *args: Any, **kwargs: Any) -> Any:
        """Rate-limited aresponse_structured."""
        await self._acquire()
        return await self._inner.aresponse_structured(*args, **kwargs)

    # -- ToolRouter protocol methods --

    async def acall_tool(self, *args: Any, **kwargs: Any) -> Any:
        """Rate-limited acall_tool."""
        await self._acquire()
        return await self._inner.acall_tool(*args, **kwargs)

    # -- Passthrough --

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the inner port."""
        return getattr(self._inner, name)
