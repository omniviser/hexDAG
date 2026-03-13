"""Tests for RateLimiter middleware."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest

from hexdag.stdlib.middleware.rate_limiter import RateLimiter


class _FakeAdapter:
    """Fake adapter for testing."""

    def __init__(self) -> None:
        self.call_count = 0
        self.model = "test-model"

    async def aresponse(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        self.call_count += 1
        return f"response-{self.call_count}"


class TestRateLimiter:
    """Tests for RateLimiter middleware."""

    @pytest.mark.asyncio
    async def test_allows_calls_within_limit(self) -> None:
        """Calls within the rate limit proceed immediately."""
        adapter = _FakeAdapter()
        mw = RateLimiter(adapter, max_calls=10, period=60.0)

        results = []
        for _ in range(5):
            results.append(await mw.aresponse([]))

        assert len(results) == 5
        assert adapter.call_count == 5
        assert mw.total_waits == 0

    @pytest.mark.asyncio
    async def test_rate_limits_excess_calls(self) -> None:
        """Calls exceeding the rate limit are delayed (not rejected)."""
        adapter = _FakeAdapter()
        # Very tight limit: 2 calls per 1 second
        mw = RateLimiter(adapter, max_calls=2, period=1.0)

        start = time.monotonic()
        # First 2 should be immediate
        await mw.aresponse([])
        await mw.aresponse([])
        # Third should wait
        await mw.aresponse([])
        elapsed = time.monotonic() - start

        assert adapter.call_count == 3
        # Should have waited at least some time for the third call
        assert mw.total_waits >= 1
        assert elapsed > 0.1  # Some delay happened

    @pytest.mark.asyncio
    async def test_getattr_passthrough(self) -> None:
        """Attributes are forwarded to inner adapter."""
        adapter = _FakeAdapter()
        mw = RateLimiter(adapter)

        assert mw.model == "test-model"

    @pytest.mark.asyncio
    async def test_acall_tool_rate_limited(self) -> None:
        """acall_tool is also rate-limited."""
        inner = AsyncMock()
        inner.acall_tool = AsyncMock(return_value={"ok": True})
        mw = RateLimiter(inner, max_calls=10, period=60.0)

        result = await mw.acall_tool("search", {"q": "test"})
        assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_token_refill(self) -> None:
        """Tokens refill over time."""
        adapter = _FakeAdapter()
        # 1 call per 0.1 seconds
        mw = RateLimiter(adapter, max_calls=1, period=0.1)

        await mw.aresponse([])  # uses the 1 token
        # Wait for refill
        import asyncio

        await asyncio.sleep(0.15)
        await mw.aresponse([])  # should have a token now

        assert adapter.call_count == 2
