"""Tests for ResponseCache middleware."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from hexdag.stdlib.middleware.response_cache import ResponseCache


class _FakeAdapter:
    """Fake adapter that tracks call count."""

    def __init__(self) -> None:
        self.call_count = 0
        self.model = "test-model"

    async def aresponse(self, messages, **kwargs):  # noqa: ANN001, ANN003, ANN201
        self.call_count += 1
        return f"response-{self.call_count}"


class TestResponseCache:
    """Tests for ResponseCache middleware."""

    @pytest.mark.asyncio
    async def test_caches_identical_calls(self) -> None:
        """Identical calls return cached result."""
        adapter = _FakeAdapter()
        mw = ResponseCache(adapter)

        r1 = await mw.aresponse("hello")
        r2 = await mw.aresponse("hello")

        assert r1 == r2 == "response-1"
        assert adapter.call_count == 1
        assert mw.hits == 1
        assert mw.misses == 1

    @pytest.mark.asyncio
    async def test_different_calls_not_cached(self) -> None:
        """Different arguments produce different cache entries."""
        adapter = _FakeAdapter()
        mw = ResponseCache(adapter)

        r1 = await mw.aresponse("hello")
        r2 = await mw.aresponse("world")

        assert r1 == "response-1"
        assert r2 == "response-2"
        assert adapter.call_count == 2
        assert mw.hits == 0
        assert mw.misses == 2

    @pytest.mark.asyncio
    async def test_lru_eviction(self) -> None:
        """Oldest entries are evicted when cache exceeds max_size."""
        adapter = _FakeAdapter()
        mw = ResponseCache(adapter, max_size=2)

        await mw.aresponse("a")
        await mw.aresponse("b")
        await mw.aresponse("c")  # evicts "a"

        assert mw.size == 2

        # "a" should be evicted, causing a new call
        await mw.aresponse("a")
        assert adapter.call_count == 4  # a, b, c, a again

    @pytest.mark.asyncio
    async def test_clear_resets_cache(self) -> None:
        """clear() empties cache and resets counters."""
        adapter = _FakeAdapter()
        mw = ResponseCache(adapter)

        await mw.aresponse("hello")
        assert mw.size == 1

        mw.clear()
        assert mw.size == 0
        assert mw.hits == 0
        assert mw.misses == 0

    @pytest.mark.asyncio
    async def test_getattr_passthrough(self) -> None:
        """Attributes are forwarded to inner adapter."""
        adapter = _FakeAdapter()
        mw = ResponseCache(adapter)

        assert mw.model == "test-model"

    @pytest.mark.asyncio
    async def test_acall_tool_cached(self) -> None:
        """acall_tool is also cached."""
        inner = AsyncMock()
        inner.acall_tool = AsyncMock(return_value={"result": "cached"})
        mw = ResponseCache(inner)

        r1 = await mw.acall_tool("search", {"q": "test"})
        r2 = await mw.acall_tool("search", {"q": "test"})

        assert r1 == r2
        assert inner.acall_tool.call_count == 1
        assert mw.hits == 1

    @pytest.mark.asyncio
    async def test_cache_size_property(self) -> None:
        """size property reflects current cache entries."""
        adapter = _FakeAdapter()
        mw = ResponseCache(adapter)

        assert mw.size == 0
        await mw.aresponse("a")
        assert mw.size == 1
        await mw.aresponse("b")
        assert mw.size == 2
        await mw.aresponse("a")  # hit, no new entry
        assert mw.size == 2
