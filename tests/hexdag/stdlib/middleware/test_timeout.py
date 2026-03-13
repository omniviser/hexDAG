"""Tests for Timeout middleware."""

from __future__ import annotations

import asyncio

import pytest

from hexdag.stdlib.middleware.timeout import Timeout


class _FakeAdapter:
    """Fake adapter that can simulate slow responses."""

    def __init__(self, delay: float = 0.0) -> None:
        self._delay = delay
        self.model = "test-model"

    async def aresponse(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        return "ok"


class TestTimeout:
    """Tests for Timeout middleware."""

    @pytest.mark.asyncio
    async def test_fast_call_succeeds(self) -> None:
        """Calls completing within timeout return normally."""
        adapter = _FakeAdapter(delay=0.0)
        mw = Timeout(adapter, timeout_seconds=1.0)

        result = await mw.aresponse([])
        assert result == "ok"
        assert mw.timeouts == 0

    @pytest.mark.asyncio
    async def test_slow_call_times_out(self) -> None:
        """Calls exceeding timeout raise TimeoutError."""
        adapter = _FakeAdapter(delay=5.0)
        mw = Timeout(adapter, timeout_seconds=0.1)

        with pytest.raises(TimeoutError):
            await mw.aresponse([])

        assert mw.timeouts == 1

    @pytest.mark.asyncio
    async def test_timeout_counter_increments(self) -> None:
        """timeouts property tracks cumulative timeout count."""
        adapter = _FakeAdapter(delay=5.0)
        mw = Timeout(adapter, timeout_seconds=0.05)

        for _ in range(3):
            with pytest.raises(TimeoutError):
                await mw.aresponse([])

        assert mw.timeouts == 3

    @pytest.mark.asyncio
    async def test_getattr_passthrough(self) -> None:
        """Attributes are forwarded to inner adapter."""
        adapter = _FakeAdapter()
        mw = Timeout(adapter, timeout_seconds=1.0)

        assert mw.model == "test-model"

    @pytest.mark.asyncio
    async def test_acall_tool_with_timeout(self) -> None:
        """acall_tool also respects timeout."""

        class SlowRouter:
            async def acall_tool(self, name, params):  # noqa: ANN001, ANN201
                await asyncio.sleep(5.0)
                return {"result": "too late"}

        mw = Timeout(SlowRouter(), timeout_seconds=0.05)

        with pytest.raises(TimeoutError):
            await mw.acall_tool("search", {"q": "test"})

        assert mw.timeouts == 1

    @pytest.mark.asyncio
    async def test_aresponse_structured_with_timeout(self) -> None:
        """aresponse_structured also respects timeout."""

        class SlowLLM:
            async def aresponse_structured(self, messages, schema):  # noqa: ANN001, ANN201
                await asyncio.sleep(5.0)
                return {}

        mw = Timeout(SlowLLM(), timeout_seconds=0.05)

        with pytest.raises(TimeoutError):
            await mw.aresponse_structured([], {})

        assert mw.timeouts == 1
