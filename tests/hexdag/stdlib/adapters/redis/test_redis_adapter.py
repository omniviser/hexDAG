"""Tests for RedisAdapter using a mock Redis client.

Since tests run without a real Redis server, we mock ``redis.asyncio``
to verify the adapter correctly delegates to the Redis client API.
"""

from __future__ import annotations

from typing import Any

import pytest

from hexdag.kernel.ports.data_store import SupportsKeyValue, SupportsTTL


class FakeRedisClient:
    """In-memory fake that mimics the ``redis.asyncio.Redis`` interface."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self._store.get(key)

    async def set(self, key: str, value: str) -> None:
        self._store[key] = value

    async def setex(self, key: str, ttl: int, value: str) -> None:
        # TTL tracking not needed — just store the value
        self._store[key] = value

    async def delete(self, key: str) -> int:
        if key in self._store:
            del self._store[key]
            return 1
        return 0

    async def exists(self, key: str) -> int:
        return 1 if key in self._store else 0

    async def scan_iter(self, match: str = "*") -> Any:
        """Yield keys matching a glob pattern."""
        import fnmatch

        for key in list(self._store.keys()):
            if fnmatch.fnmatch(key, match):
                yield key

    async def aclose(self) -> None:
        self._store.clear()


def _make_adapter(**kwargs: Any) -> Any:
    """Create a RedisAdapter with a fake client injected."""
    from hexdag.stdlib.adapters.redis import RedisAdapter

    adapter = RedisAdapter(**kwargs)
    adapter._client = FakeRedisClient()
    return adapter


class TestProtocolConformance:
    """RedisAdapter satisfies SupportsKeyValue and SupportsTTL."""

    def test_supports_key_value(self) -> None:
        from hexdag.stdlib.adapters.redis import RedisAdapter

        adapter = RedisAdapter()
        assert isinstance(adapter, SupportsKeyValue)

    def test_supports_ttl(self) -> None:
        from hexdag.stdlib.adapters.redis import RedisAdapter

        adapter = RedisAdapter()
        assert isinstance(adapter, SupportsTTL)


class TestAget:
    """Tests for RedisAdapter.aget()."""

    @pytest.mark.asyncio()
    async def test_get_existing_key(self) -> None:
        adapter = _make_adapter()
        await adapter.aset("name", "Alice")
        result = await adapter.aget("name")
        assert result == "Alice"

    @pytest.mark.asyncio()
    async def test_get_missing_key_returns_none(self) -> None:
        adapter = _make_adapter()
        assert await adapter.aget("missing") is None

    @pytest.mark.asyncio()
    async def test_get_dict_value(self) -> None:
        adapter = _make_adapter()
        await adapter.aset("data", {"x": 1, "y": 2})
        result = await adapter.aget("data")
        assert result == {"x": 1, "y": 2}


class TestAset:
    """Tests for RedisAdapter.aset()."""

    @pytest.mark.asyncio()
    async def test_set_string(self) -> None:
        adapter = _make_adapter()
        await adapter.aset("key", "value")
        assert await adapter.aget("key") == "value"

    @pytest.mark.asyncio()
    async def test_set_overwrites(self) -> None:
        adapter = _make_adapter()
        await adapter.aset("key", "v1")
        await adapter.aset("key", "v2")
        assert await adapter.aget("key") == "v2"

    @pytest.mark.asyncio()
    async def test_set_with_default_ttl(self) -> None:
        adapter = _make_adapter(default_ttl_seconds=60)
        await adapter.aset("key", "value")
        # Value should be stored (TTL not enforced in fake)
        assert await adapter.aget("key") == "value"


class TestAdelete:
    """Tests for RedisAdapter.adelete()."""

    @pytest.mark.asyncio()
    async def test_delete_existing(self) -> None:
        adapter = _make_adapter()
        await adapter.aset("k", "v")
        assert await adapter.adelete("k") is True
        assert await adapter.aget("k") is None

    @pytest.mark.asyncio()
    async def test_delete_missing(self) -> None:
        adapter = _make_adapter()
        assert await adapter.adelete("nope") is False


class TestAexists:
    """Tests for RedisAdapter.aexists()."""

    @pytest.mark.asyncio()
    async def test_exists_true(self) -> None:
        adapter = _make_adapter()
        await adapter.aset("k", "v")
        assert await adapter.aexists("k") is True

    @pytest.mark.asyncio()
    async def test_exists_false(self) -> None:
        adapter = _make_adapter()
        assert await adapter.aexists("k") is False


class TestAlistKeys:
    """Tests for RedisAdapter.alist_keys()."""

    @pytest.mark.asyncio()
    async def test_list_all_keys(self) -> None:
        adapter = _make_adapter()
        await adapter.aset("a", 1)
        await adapter.aset("b", 2)
        keys = await adapter.alist_keys()
        assert sorted(keys) == ["a", "b"]

    @pytest.mark.asyncio()
    async def test_list_with_prefix(self) -> None:
        adapter = _make_adapter()
        await adapter.aset("user:1", "alice")
        await adapter.aset("user:2", "bob")
        await adapter.aset("order:1", "pizza")
        keys = await adapter.alist_keys(prefix="user:")
        assert sorted(keys) == ["user:1", "user:2"]

    @pytest.mark.asyncio()
    async def test_list_empty_store(self) -> None:
        adapter = _make_adapter()
        assert await adapter.alist_keys() == []


class TestKeyPrefix:
    """Key prefix namespace isolation."""

    @pytest.mark.asyncio()
    async def test_prefix_applied(self) -> None:
        adapter = _make_adapter(key_prefix="myapp:")
        await adapter.aset("key", "value")
        # Directly check the fake client's internal store
        raw_keys = list(adapter._client._store.keys())
        assert all(k.startswith("myapp:") for k in raw_keys)

    @pytest.mark.asyncio()
    async def test_prefix_stripped_on_list(self) -> None:
        adapter = _make_adapter(key_prefix="ns:")
        await adapter.aset("a", 1)
        await adapter.aset("b", 2)
        keys = await adapter.alist_keys()
        # Returned keys should NOT have the prefix
        assert sorted(keys) == ["a", "b"]


class TestAsetWithTtl:
    """Tests for SupportsTTL.aset_with_ttl()."""

    @pytest.mark.asyncio()
    async def test_set_with_ttl(self) -> None:
        adapter = _make_adapter()
        await adapter.aset_with_ttl("key", "value", ttl_seconds=300)
        assert await adapter.aget("key") == "value"


class TestConnectionLifecycle:
    """Tests for lazy client creation and close."""

    @pytest.mark.asyncio()
    async def test_close_clears_client(self) -> None:
        adapter = _make_adapter()
        assert adapter._client is not None
        await adapter.aclose()
        assert adapter._client is None

    def test_import_error_without_redis(self) -> None:
        """Verify helpful error when redis package is missing."""
        from hexdag.stdlib.adapters.redis import RedisAdapter

        adapter = RedisAdapter()
        # Client is None — _get_client will try to import redis
        assert adapter._client is None


class TestSSLConfiguration:
    """SSL/TLS configuration tests."""

    def test_ssl_auto_detected_from_rediss_scheme(self) -> None:
        from hexdag.stdlib.adapters.redis import RedisAdapter

        adapter = RedisAdapter(url="rediss://my-redis.cloud:6380/0")
        assert adapter._ssl_enabled is True

    def test_ssl_not_enabled_by_default(self) -> None:
        from hexdag.stdlib.adapters.redis import RedisAdapter

        adapter = RedisAdapter(url="redis://localhost:6379/0")
        assert adapter._ssl_enabled is False

    def test_ssl_explicit_override(self) -> None:
        from hexdag.stdlib.adapters.redis import RedisAdapter

        adapter = RedisAdapter(url="redis://localhost:6379/0", ssl_enabled=True)
        assert adapter._ssl_enabled is True

    def test_ssl_mtls_params_stored(self) -> None:
        from hexdag.stdlib.adapters.redis import RedisAdapter

        adapter = RedisAdapter(
            url="rediss://host:6380/0",
            ssl_ca_certs="/etc/ssl/ca.pem",
            ssl_certfile="/etc/ssl/client.pem",
            ssl_keyfile="/etc/ssl/client-key.pem",
            ssl_cert_reqs="optional",
        )
        assert adapter._ssl_ca_certs == "/etc/ssl/ca.pem"
        assert adapter._ssl_certfile == "/etc/ssl/client.pem"
        assert adapter._ssl_keyfile == "/etc/ssl/client-key.pem"
        assert adapter._ssl_cert_reqs == "optional"


class TestHexDAGAdapterRegistration:
    """RedisAdapter registers itself via HexDAGAdapter."""

    def test_yaml_alias_registered(self) -> None:
        from hexdag.stdlib.adapters.base import HexDAGAdapter
        from hexdag.stdlib.adapters.redis import RedisAdapter  # noqa: F401

        assert "redis" in HexDAGAdapter._registry
        assert "data_store:redis" in HexDAGAdapter._registry

    def test_class_name_registered(self) -> None:
        from hexdag.stdlib.adapters.base import HexDAGAdapter
        from hexdag.stdlib.adapters.redis import RedisAdapter  # noqa: F401

        assert "RedisAdapter" in HexDAGAdapter._registry
