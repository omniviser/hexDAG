"""Distributed cache middleware.

Caches port call results in an external store (Redis, SQLite, any
``SupportsKeyValue + SupportsTTL`` adapter) to share cached responses
across pipeline instances and process restarts.

Unlike ``ResponseCache`` which uses in-process LRU memory, this
middleware delegates storage to a pluggable ``DataStore`` adapter,
enabling distributed and persistent caching.

Example YAML::

    spec:
      ports:
        cache_store:
          adapter: hexdag.stdlib.adapters.redis.RedisAdapter
          config:
            url: redis://localhost:6379/0
            key_prefix: "cache:"

        llm:
          adapter: hexdag.stdlib.adapters.openai.OpenAIAdapter
          middleware:
            - class: hexdag.stdlib.middleware.distributed_cache.DistributedCache
              config:
                store: ref:cache_store
                ttl_seconds: 3600
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from hexdag.kernel.logging import get_logger
from hexdag.kernel.ports.data_store import SupportsKeyValue, SupportsTTL

logger = get_logger(__name__)

# Default TTL: 1 hour
_DEFAULT_TTL_SECONDS = 3600
_CACHE_KEY_PREFIX = "dcache:"


class DistributedCache:
    """Middleware that caches port call responses in an external store.

    The store must implement ``SupportsKeyValue`` and optionally
    ``SupportsTTL`` for automatic expiry.  If the store does not
    support TTL, entries persist until manually evicted.

    Parameters
    ----------
    inner : Any
        The port/adapter to wrap.
    store : SupportsKeyValue
        The external cache store (e.g. ``RedisAdapter``, ``InMemoryMemory``).
    ttl_seconds : int
        Time-to-live for cached entries in seconds (default 3600).
        Only applied if the store implements ``SupportsTTL``.
    key_prefix : str
        Prefix for all cache keys (default ``"dcache:"``).
    """

    def __init__(
        self,
        inner: Any,
        store: SupportsKeyValue,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
        key_prefix: str = _CACHE_KEY_PREFIX,
    ) -> None:
        self._inner = inner
        self._store: SupportsKeyValue = store
        self._ttl_seconds = ttl_seconds
        self._key_prefix = key_prefix
        self._hits = 0
        self._misses = 0

        # Check TTL support once at init
        self._supports_ttl = isinstance(store, SupportsTTL)
        self._ttl_store: SupportsTTL | None = store if isinstance(store, SupportsTTL) else None

    @property
    def hits(self) -> int:
        """Total cache hits."""
        return self._hits

    @property
    def misses(self) -> int:
        """Total cache misses."""
        return self._misses

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
            serialised = repr((method_name, args, kwargs))
        digest = hashlib.sha256(serialised.encode()).hexdigest()
        return f"{self._key_prefix}{digest}"

    async def _cached_call(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Call a method with distributed caching."""
        key = self._make_key(method_name, args, kwargs)

        # Try cache lookup
        cached = await self._store.aget(key)
        if cached is not None:
            self._hits += 1
            logger.debug("Distributed cache hit for {}", method_name)
            return cached

        # Cache miss — call inner
        self._misses += 1
        method = getattr(self._inner, method_name)
        result = await method(*args, **kwargs)

        # Store result
        if self._ttl_store is not None:
            await self._ttl_store.aset_with_ttl(key, result, self._ttl_seconds)
        else:
            await self._store.aset(key, result)

        return result

    async def invalidate(self, method_name: str, *args: Any, **kwargs: Any) -> bool:
        """Invalidate a specific cached entry.

        Pass the same method name and arguments as the original call.
        Returns ``True`` if the entry existed and was removed.
        """
        key = self._make_key(method_name, args, kwargs)
        return await self._store.adelete(key)

    async def clear(self) -> None:
        """Clear all distributed cache entries with this prefix.

        Uses ``alist_keys`` to find and delete all matching keys.
        """
        keys = await self._store.alist_keys(prefix=self._key_prefix)
        for key in keys:
            await self._store.adelete(key)
        self._hits = 0
        self._misses = 0

    # -- LLM protocol methods -------------------------------------------------

    async def aresponse(self, *args: Any, **kwargs: Any) -> Any:
        """Cached aresponse."""
        return await self._cached_call("aresponse", *args, **kwargs)

    async def aresponse_with_tools(self, *args: Any, **kwargs: Any) -> Any:
        """Cached aresponse_with_tools."""
        return await self._cached_call("aresponse_with_tools", *args, **kwargs)

    async def aresponse_structured(self, *args: Any, **kwargs: Any) -> Any:
        """Cached aresponse_structured."""
        return await self._cached_call("aresponse_structured", *args, **kwargs)

    # -- ToolRouter protocol methods ------------------------------------------

    async def acall_tool(self, *args: Any, **kwargs: Any) -> Any:
        """Cached acall_tool."""
        return await self._cached_call("acall_tool", *args, **kwargs)

    # -- Passthrough ----------------------------------------------------------

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
