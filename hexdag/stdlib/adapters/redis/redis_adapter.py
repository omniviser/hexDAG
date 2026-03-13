"""Redis-backed DataStore adapter.

Implements ``SupportsKeyValue`` and ``SupportsTTL`` using the
``redis.asyncio`` client.  All keys are optionally prefixed with
``key_prefix`` to allow namespace isolation within a shared Redis
instance.

Example::

    # Plain Redis
    adapter = RedisAdapter(url="redis://localhost:6379/0", key_prefix="myapp:")

    # Redis with TLS (auto-detected from rediss:// scheme)
    adapter = RedisAdapter(url="rediss://my-redis.cloud:6380/0")

    # Redis with mutual TLS (mTLS)
    adapter = RedisAdapter(
        url="rediss://my-redis.cloud:6380/0",
        ssl_ca_certs="/etc/ssl/ca.pem",
        ssl_certfile="/etc/ssl/client.pem",
        ssl_keyfile="/etc/ssl/client-key.pem",
    )
"""

from __future__ import annotations

import json
import ssl
from typing import TYPE_CHECKING, Any

from hexdag.kernel.logging import get_logger
from hexdag.kernel.ports.data_store import SupportsKeyValue, SupportsTTL
from hexdag.stdlib.adapters.base import HexDAGAdapter

if TYPE_CHECKING:
    import redis.asyncio as aioredis  # pyright: ignore[reportMissingImports]

logger = get_logger(__name__)

__all__ = ["RedisAdapter"]

# Sentinel for distinguishing "key not found" from stored None
_MISSING = object()


class RedisAdapter(
    HexDAGAdapter,
    SupportsKeyValue,
    SupportsTTL,
    yaml_alias="redis",
    port="data_store",
):
    """Redis-backed key-value store with TTL support.

    Parameters
    ----------
    url : str
        Redis connection URL.  Use ``rediss://`` scheme for TLS
        (e.g. ``rediss://my-redis.cloud:6380/0``).
    key_prefix : str
        Optional prefix prepended to all keys for namespace isolation.
    decode_responses : bool
        Whether Redis should decode bytes to str (default True).
    default_ttl_seconds : int | None
        Optional default TTL applied to all ``aset`` calls.
        ``None`` means no automatic expiry.
    ssl_enabled : bool
        Explicitly enable TLS.  Auto-detected when the URL scheme is
        ``rediss://``, but can be forced for non-standard setups.
    ssl_ca_certs : str | None
        Path to a CA bundle file for server certificate verification.
    ssl_certfile : str | None
        Path to a client certificate for mutual TLS (mTLS).
    ssl_keyfile : str | None
        Path to the client private key for mTLS.
    ssl_cert_reqs : str
        Certificate verification mode: ``"required"`` (default),
        ``"optional"``, or ``"none"``.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        key_prefix: str = "",
        decode_responses: bool = True,
        default_ttl_seconds: int | None = None,
        ssl_enabled: bool = False,
        ssl_ca_certs: str | None = None,
        ssl_certfile: str | None = None,
        ssl_keyfile: str | None = None,
        ssl_cert_reqs: str = "required",
        **kwargs: Any,
    ) -> None:
        self._url = url
        self._key_prefix = key_prefix
        self._decode_responses = decode_responses
        self._default_ttl_seconds = default_ttl_seconds
        self._ssl_enabled = ssl_enabled or url.startswith("rediss://")
        self._ssl_ca_certs = ssl_ca_certs
        self._ssl_certfile = ssl_certfile
        self._ssl_keyfile = ssl_keyfile
        self._ssl_cert_reqs = ssl_cert_reqs
        self._client: aioredis.Redis | None = None

    # -- Connection lifecycle -------------------------------------------------

    async def _get_client(self) -> aioredis.Redis:
        """Lazily create and return the async Redis client."""
        if self._client is None:
            try:
                import redis.asyncio as aioredis  # lazy: optional  # pyright: ignore[reportMissingImports]  # noqa: E501
            except ImportError as exc:
                msg = (
                    "RedisAdapter requires the 'redis' package. Install it with: pip install redis"
                )
                raise ImportError(msg) from exc

            connection_kwargs: dict[str, Any] = {
                "decode_responses": self._decode_responses,
            }

            if self._ssl_enabled:
                ctx = ssl.create_default_context(cafile=self._ssl_ca_certs)
                if self._ssl_certfile:
                    ctx.load_cert_chain(
                        certfile=self._ssl_certfile,
                        keyfile=self._ssl_keyfile,
                    )
                _CERT_REQS_MAP = {
                    "required": ssl.CERT_REQUIRED,
                    "optional": ssl.CERT_OPTIONAL,
                    "none": ssl.CERT_NONE,
                }
                ctx.check_hostname = self._ssl_cert_reqs == "required"
                ctx.verify_mode = _CERT_REQS_MAP.get(self._ssl_cert_reqs, ssl.CERT_REQUIRED)
                connection_kwargs["ssl"] = ctx

            self._client = aioredis.from_url(
                self._url,
                **connection_kwargs,
            )
        return self._client

    async def aclose(self) -> None:
        """Close the Redis connection."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- Key helpers ----------------------------------------------------------

    def _prefixed(self, key: str) -> str:
        """Return the prefixed key for Redis storage."""
        return f"{self._key_prefix}{key}"

    # -- SupportsKeyValue -----------------------------------------------------

    async def aget(self, key: str) -> Any:
        """Retrieve a value by key. Returns ``None`` when absent."""
        client = await self._get_client()
        raw = await client.get(self._prefixed(key))
        if raw is None:
            return None
        return _deserialise(raw)

    async def aset(self, key: str, value: Any) -> None:
        """Store a value under key (upsert). Applies default TTL if configured."""
        client = await self._get_client()
        serialised = _serialise(value)
        if self._default_ttl_seconds is not None:
            await client.setex(
                self._prefixed(key),
                self._default_ttl_seconds,
                serialised,
            )
        else:
            await client.set(self._prefixed(key), serialised)

    async def adelete(self, key: str) -> bool:
        """Delete a key. Returns ``True`` if it existed."""
        client = await self._get_client()
        removed = await client.delete(self._prefixed(key))
        return bool(removed > 0)

    async def aexists(self, key: str) -> bool:
        """Check whether a key exists."""
        client = await self._get_client()
        return bool(await client.exists(self._prefixed(key)))

    async def alist_keys(self, prefix: str = "") -> list[str]:
        """List keys matching an optional prefix.

        Scans using the Redis ``SCAN`` command to avoid blocking on
        large keyspaces.  The returned keys have ``key_prefix`` stripped.
        """
        client = await self._get_client()
        pattern = f"{self._key_prefix}{prefix}*"
        keys: list[str] = []
        async for key in client.scan_iter(match=pattern):
            # Strip the adapter-level prefix to return logical keys
            k = key if isinstance(key, str) else key.decode()
            if self._key_prefix and k.startswith(self._key_prefix):
                k = k[len(self._key_prefix) :]
            keys.append(k)
        return keys

    # -- SupportsTTL ----------------------------------------------------------

    async def aset_with_ttl(self, key: str, value: Any, ttl_seconds: int) -> None:
        """Store a value with explicit TTL expiry."""
        client = await self._get_client()
        serialised = _serialise(value)
        await client.setex(self._prefixed(key), ttl_seconds, serialised)


# -- Serialisation helpers ----------------------------------------------------


def _serialise(value: Any) -> str:
    """Serialise a Python object to a JSON string for Redis storage."""
    return json.dumps(value, default=str, sort_keys=True)


def _deserialise(raw: str | bytes) -> Any:
    """Deserialise a JSON string from Redis back to a Python object."""
    text = raw if isinstance(raw, str) else raw.decode()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text
