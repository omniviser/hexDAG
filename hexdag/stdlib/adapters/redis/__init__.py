"""Redis adapter for hexDAG DataStore port.

Implements ``SupportsKeyValue`` and ``SupportsTTL`` backed by Redis,
enabling distributed caching, shared state, and TTL-based expiry
across multiple pipeline instances.

Requires the ``redis`` package (``pip install redis``).

Example YAML::

    spec:
      ports:
        cache_store:
          adapter: hexdag.stdlib.adapters.redis.RedisAdapter
          config:
            url: redis://localhost:6379/0
            key_prefix: "hexdag:"
"""

from hexdag.stdlib.adapters.redis.redis_adapter import RedisAdapter

__all__ = ["RedisAdapter"]
