"""Response cache middleware.

Caches deterministic port call results in memory to avoid redundant API calls.
Uses a hash of the method name and arguments as the cache key.

Example YAML::

    spec:
      ports:
        llm:
          adapter: hexdag.stdlib.adapters.openai.OpenAIAdapter
          middleware:
            - hexdag.stdlib.middleware.response_cache.ResponseCache
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from typing import Any

from hexdag.kernel.logging import get_logger

logger = get_logger(__name__)

# Default configuration
_DEFAULT_MAX_SIZE = 128
_SENTINEL = object()


class ResponseCache:
    """Middleware that caches port call responses in an LRU cache.

    Identical calls (same method + arguments) return cached results
    without hitting the underlying adapter.  Uses an ordered dict
    as a simple LRU eviction strategy.

    Parameters
    ----------
    inner : Any
        The port/adapter to wrap.
    max_size : int
        Maximum number of cached responses (default 128).
        Oldest entries are evicted when full.
    """

    def __init__(
        self,
        inner: Any,
        max_size: int = _DEFAULT_MAX_SIZE,
    ) -> None:
        """Initialize response cache with LRU eviction."""
        self._inner = inner
        self._max_size = max_size
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._hits = 0
        self._misses = 0

    @property
    def hits(self) -> int:
        """Total cache hits."""
        return self._hits

    @property
    def misses(self) -> int:
        """Total cache misses."""
        return self._misses

    @property
    def size(self) -> int:
        """Current number of cached entries."""
        return len(self._cache)

    def _make_key(self, method_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        """Create a deterministic cache key from method + arguments."""
        try:
            payload = {
                "method": method_name,
                "args": _serialisable(args),
                "kwargs": _serialisable(kwargs),
            }
            serialised = json.dumps(payload, sort_keys=True, default=str)
        except (TypeError, ValueError):
            # Fallback: use repr for non-serialisable args
            serialised = repr((method_name, args, kwargs))
        return hashlib.sha256(serialised.encode()).hexdigest()

    async def _cached_call(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Call a method with caching."""
        key = self._make_key(method_name, args, kwargs)

        if key in self._cache:
            self._hits += 1
            self._cache.move_to_end(key)
            logger.debug("Cache hit for {}", method_name)
            return self._cache[key]

        self._misses += 1
        method = getattr(self._inner, method_name)
        result = await method(*args, **kwargs)

        # Store in cache
        self._cache[key] = result
        self._cache.move_to_end(key)

        # Evict oldest if over limit
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

        return result

    def clear(self) -> None:
        """Clear the cache and reset counters."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    # -- LLM protocol methods --

    async def aresponse(self, *args: Any, **kwargs: Any) -> Any:
        """Cached aresponse."""
        return await self._cached_call("aresponse", *args, **kwargs)

    async def aresponse_with_tools(self, *args: Any, **kwargs: Any) -> Any:
        """Cached aresponse_with_tools."""
        return await self._cached_call("aresponse_with_tools", *args, **kwargs)

    async def aresponse_structured(self, *args: Any, **kwargs: Any) -> Any:
        """Cached aresponse_structured."""
        return await self._cached_call("aresponse_structured", *args, **kwargs)

    # -- ToolRouter protocol methods --

    async def acall_tool(self, *args: Any, **kwargs: Any) -> Any:
        """Cached acall_tool."""
        return await self._cached_call("acall_tool", *args, **kwargs)

    # -- Passthrough --

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the inner port."""
        return getattr(self._inner, name)


def _serialisable(obj: Any) -> Any:
    """Convert objects to JSON-serialisable form."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {str(k): _serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialisable(item) for item in obj]
    if hasattr(obj, "__dict__"):
        attrs = {k: _serialisable(v) for k, v in obj.__dict__.items()}
        return {"__type__": type(obj).__name__, **attrs}
    return str(obj)
