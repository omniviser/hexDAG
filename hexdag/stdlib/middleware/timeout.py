"""Timeout middleware.

Wraps any port and enforces a maximum wall-clock duration per call.
Raises ``asyncio.TimeoutError`` if the call exceeds the limit.

Example YAML::

    spec:
      ports:
        llm:
          adapter: hexdag.stdlib.adapters.openai.OpenAIAdapter
          middleware:
            - hexdag.stdlib.middleware.timeout.Timeout
"""

from __future__ import annotations

import asyncio
from typing import Any

from hexdag.kernel.logging import get_logger

logger = get_logger(__name__)

# Default configuration
_DEFAULT_TIMEOUT = 30.0


class Timeout:
    """Middleware that enforces a maximum duration per port call.

    If the inner port does not respond within ``timeout_seconds``,
    ``asyncio.TimeoutError`` is raised.

    Parameters
    ----------
    inner : Any
        The port/adapter to wrap.
    timeout_seconds : float
        Maximum allowed duration per call in seconds (default 30.0).
    """

    def __init__(
        self,
        inner: Any,
        timeout_seconds: float = _DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize timeout middleware with duration limit."""
        self._inner = inner
        self._timeout_seconds = timeout_seconds
        self._timeouts = 0

    @property
    def timeouts(self) -> int:
        """Total number of calls that timed out."""
        return self._timeouts

    async def _call_with_timeout(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Call a method on the inner port with timeout enforcement."""
        method = getattr(self._inner, method_name)
        try:
            return await asyncio.wait_for(
                method(*args, **kwargs),
                timeout=self._timeout_seconds,
            )
        except TimeoutError:
            self._timeouts += 1
            logger.warning(
                "Timeout ({:.1f}s) exceeded for {}.{}",
                self._timeout_seconds,
                type(self._inner).__name__,
                method_name,
            )
            raise

    # -- LLM protocol methods --

    async def aresponse(self, *args: Any, **kwargs: Any) -> Any:
        """Timeout-wrapped aresponse."""
        return await self._call_with_timeout("aresponse", *args, **kwargs)

    async def aresponse_with_tools(self, *args: Any, **kwargs: Any) -> Any:
        """Timeout-wrapped aresponse_with_tools."""
        return await self._call_with_timeout("aresponse_with_tools", *args, **kwargs)

    async def aresponse_structured(self, *args: Any, **kwargs: Any) -> Any:
        """Timeout-wrapped aresponse_structured."""
        return await self._call_with_timeout("aresponse_structured", *args, **kwargs)

    # -- ToolRouter protocol methods --

    async def acall_tool(self, *args: Any, **kwargs: Any) -> Any:
        """Timeout-wrapped acall_tool."""
        return await self._call_with_timeout("acall_tool", *args, **kwargs)

    # -- Passthrough --

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the inner port."""
        return getattr(self._inner, name)
