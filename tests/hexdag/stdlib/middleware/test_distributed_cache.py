"""Tests for DistributedCache middleware.

Uses ``InMemoryMemory`` as the backing store — no external dependencies.
"""

from __future__ import annotations

from typing import Any

import pytest

from hexdag.stdlib.adapters.memory import InMemoryMemory
from hexdag.stdlib.middleware.distributed_cache import DistributedCache


class FakePort:
    """Minimal fake port that records calls and returns canned responses."""

    def __init__(self, response: Any = "mock-response") -> None:
        self._response = response
        self.call_count = 0
        self.last_args: tuple[Any, ...] = ()
        self.last_kwargs: dict[str, Any] = {}

    async def aresponse(self, *args: Any, **kwargs: Any) -> Any:
        self.call_count += 1
        self.last_args = args
        self.last_kwargs = kwargs
        return self._response

    async def aresponse_with_tools(self, *args: Any, **kwargs: Any) -> Any:
        self.call_count += 1
        return self._response

    async def aresponse_structured(self, *args: Any, **kwargs: Any) -> Any:
        self.call_count += 1
        return self._response

    async def acall_tool(self, *args: Any, **kwargs: Any) -> Any:
        self.call_count += 1
        return self._response

    @property
    def model_name(self) -> str:
        return "fake-model"


class TestCacheHitMiss:
    """Basic hit/miss behaviour."""

    @pytest.mark.asyncio()
    async def test_first_call_is_miss(self) -> None:
        port = FakePort(response="hello")
        store = InMemoryMemory()
        cache = DistributedCache(port, store=store)

        result = await cache.aresponse("prompt")
        assert result == "hello"
        assert cache.misses == 1
        assert cache.hits == 0
        assert port.call_count == 1

    @pytest.mark.asyncio()
    async def test_second_call_is_hit(self) -> None:
        port = FakePort(response="hello")
        store = InMemoryMemory()
        cache = DistributedCache(port, store=store)

        await cache.aresponse("prompt")
        result = await cache.aresponse("prompt")
        assert result == "hello"
        assert cache.hits == 1
        assert cache.misses == 1
        # Inner port called only once
        assert port.call_count == 1

    @pytest.mark.asyncio()
    async def test_different_args_are_separate_entries(self) -> None:
        port = FakePort(response="r")
        store = InMemoryMemory()
        cache = DistributedCache(port, store=store)

        await cache.aresponse("a")
        await cache.aresponse("b")
        assert cache.misses == 2
        assert port.call_count == 2


class TestAllProtocolMethods:
    """All LLM + ToolRouter protocol methods use caching."""

    @pytest.mark.asyncio()
    async def test_aresponse_with_tools_cached(self) -> None:
        port = FakePort(response="tools-result")
        store = InMemoryMemory()
        cache = DistributedCache(port, store=store)

        await cache.aresponse_with_tools("p")
        result = await cache.aresponse_with_tools("p")
        assert result == "tools-result"
        assert port.call_count == 1

    @pytest.mark.asyncio()
    async def test_aresponse_structured_cached(self) -> None:
        port = FakePort(response={"key": "val"})
        store = InMemoryMemory()
        cache = DistributedCache(port, store=store)

        await cache.aresponse_structured("p")
        result = await cache.aresponse_structured("p")
        assert result == {"key": "val"}
        assert port.call_count == 1

    @pytest.mark.asyncio()
    async def test_acall_tool_cached(self) -> None:
        port = FakePort(response="tool-out")
        store = InMemoryMemory()
        cache = DistributedCache(port, store=store)

        await cache.acall_tool("fn", arg="x")
        result = await cache.acall_tool("fn", arg="x")
        assert result == "tool-out"
        assert port.call_count == 1


class TestInvalidation:
    """Cache invalidation and clearing."""

    @pytest.mark.asyncio()
    async def test_invalidate_specific_entry(self) -> None:
        port = FakePort(response="v1")
        store = InMemoryMemory()
        cache = DistributedCache(port, store=store)

        await cache.aresponse("prompt")
        assert cache.misses == 1

        removed = await cache.invalidate("aresponse", "prompt")
        assert removed is True

        # Next call should miss again
        await cache.aresponse("prompt")
        assert cache.misses == 2

    @pytest.mark.asyncio()
    async def test_invalidate_missing_entry(self) -> None:
        port = FakePort()
        store = InMemoryMemory()
        cache = DistributedCache(port, store=store)

        removed = await cache.invalidate("aresponse", "never-called")
        assert removed is False

    @pytest.mark.asyncio()
    async def test_clear_resets_all(self) -> None:
        port = FakePort(response="v")
        store = InMemoryMemory()
        cache = DistributedCache(port, store=store)

        await cache.aresponse("a")
        await cache.aresponse("b")
        assert cache.misses == 2

        await cache.clear()
        assert cache.hits == 0
        assert cache.misses == 0

        # After clear, calls should miss
        await cache.aresponse("a")
        assert cache.misses == 1
        assert port.call_count == 3


class TestTTLSupport:
    """TTL detection and usage."""

    @pytest.mark.asyncio()
    async def test_store_without_ttl_still_works(self) -> None:
        """A store that only has SupportsKeyValue (no SupportsTTL) works."""

        class KeyValueOnly:
            """Minimal SupportsKeyValue without SupportsTTL."""

            def __init__(self) -> None:
                self._data: dict[str, Any] = {}

            async def aget(self, key: str) -> Any:
                return self._data.get(key)

            async def aset(self, key: str, value: Any) -> None:
                self._data[key] = value

            async def adelete(self, key: str) -> bool:
                existed = key in self._data
                self._data.pop(key, None)
                return existed

            async def aexists(self, key: str) -> bool:
                return key in self._data

            async def alist_keys(self, prefix: str = "") -> list[str]:
                return (
                    [k for k in self._data if k.startswith(prefix)] if prefix else list(self._data)
                )

        port = FakePort(response="no-ttl")
        store = KeyValueOnly()
        cache = DistributedCache(port, store=store)

        assert cache._supports_ttl is False
        result = await cache.aresponse("p")
        assert result == "no-ttl"

    @pytest.mark.asyncio()
    async def test_in_memory_memory_has_no_ttl(self) -> None:
        """InMemoryMemory only has SupportsKeyValue, not SupportsTTL."""
        port = FakePort()
        store = InMemoryMemory()
        cache = DistributedCache(port, store=store)
        # InMemoryMemory doesn't implement SupportsTTL
        assert cache._supports_ttl is False


class TestKeyPrefix:
    """Custom key prefix configuration."""

    @pytest.mark.asyncio()
    async def test_custom_prefix(self) -> None:
        port = FakePort(response="pfx")
        store = InMemoryMemory()
        cache = DistributedCache(port, store=store, key_prefix="myapp:")

        await cache.aresponse("test")
        # Check that stored keys start with our prefix
        keys = await store.alist_keys(prefix="myapp:")
        assert len(keys) == 1
        assert keys[0].startswith("myapp:")

    @pytest.mark.asyncio()
    async def test_different_prefixes_are_isolated(self) -> None:
        port = FakePort(response="r")
        store = InMemoryMemory()
        cache_a = DistributedCache(port, store=store, key_prefix="a:")
        cache_b = DistributedCache(port, store=store, key_prefix="b:")

        await cache_a.aresponse("prompt")
        await cache_b.aresponse("prompt")

        # Both should miss — different prefixes
        assert cache_a.misses == 1
        assert cache_b.misses == 1


class TestPassthrough:
    """Attribute passthrough to inner port."""

    @pytest.mark.asyncio()
    async def test_passthrough_attribute(self) -> None:
        port = FakePort()
        store = InMemoryMemory()
        cache = DistributedCache(port, store=store)

        assert cache.model_name == "fake-model"


class TestSharedStore:
    """Two DistributedCache instances sharing the same store."""

    @pytest.mark.asyncio()
    async def test_shared_store_cross_instance_hit(self) -> None:
        """When two caches share the same store and prefix, one's write is the other's hit."""
        port_a = FakePort(response="shared")
        port_b = FakePort(response="should-not-be-called")
        store = InMemoryMemory()

        cache_a = DistributedCache(port_a, store=store, key_prefix="shared:")
        cache_b = DistributedCache(port_b, store=store, key_prefix="shared:")

        # Cache A writes
        await cache_a.aresponse("prompt")
        assert port_a.call_count == 1

        # Cache B reads from the same store — should hit
        result = await cache_b.aresponse("prompt")
        assert result == "shared"
        assert port_b.call_count == 0
        assert cache_b.hits == 1


class TestKeyDeterminism:
    """Cache keys are deterministic for identical inputs."""

    def test_same_inputs_same_key(self) -> None:
        store = InMemoryMemory()
        cache = DistributedCache(FakePort(), store=store)

        key_a = cache._make_key("aresponse", ("hello",), {})
        key_b = cache._make_key("aresponse", ("hello",), {})
        assert key_a == key_b

    def test_different_inputs_different_key(self) -> None:
        store = InMemoryMemory()
        cache = DistributedCache(FakePort(), store=store)

        key_a = cache._make_key("aresponse", ("hello",), {})
        key_b = cache._make_key("aresponse", ("world",), {})
        assert key_a != key_b
