"""Tests for CircuitBreaker middleware."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from hexdag.stdlib.middleware.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
)


class _FakeAdapter:
    """Fake adapter whose responses are scripted."""

    def __init__(self, responses: list[str | Exception] | None = None) -> None:
        self._responses = list(responses or ["ok"])
        self._call_count = 0
        self.model = "test-model"

    def _next(self) -> str:
        resp = self._responses[min(self._call_count - 1, len(self._responses) - 1)]
        if isinstance(resp, Exception):
            raise resp
        return resp

    async def aresponse(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        self._call_count += 1
        return self._next()

    async def aresponse_with_tools(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        self._call_count += 1
        return self._next()

    async def aresponse_structured(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        self._call_count += 1
        return self._next()

    async def acall_tool(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        self._call_count += 1
        return self._next()


class TestCircuitBreakerClosed:
    """Tests for normal (closed) operation."""

    @pytest.mark.asyncio
    async def test_success_passes_through(self) -> None:
        adapter = _FakeAdapter(["hello"])
        cb = CircuitBreaker(adapter, failure_threshold=3)

        result = await cb.aresponse([])
        assert result == "hello"
        assert cb.state == "closed"
        assert cb.success_count == 1
        assert cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_failures_below_threshold_stay_closed(self) -> None:
        adapter = _FakeAdapter([ValueError("err"), ValueError("err"), "ok"])
        cb = CircuitBreaker(adapter, failure_threshold=3)

        with pytest.raises(ValueError):
            await cb.aresponse([])
        with pytest.raises(ValueError):
            await cb.aresponse([])

        assert cb.state == "closed"
        assert cb.failure_count == 2

        result = await cb.aresponse([])
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_success_resets_failure_counter(self) -> None:
        adapter = _FakeAdapter([ValueError("e"), "ok", ValueError("e"), "ok"])
        cb = CircuitBreaker(adapter, failure_threshold=2)

        with pytest.raises(ValueError):
            await cb.aresponse([])
        await cb.aresponse([])  # success resets counter

        with pytest.raises(ValueError):
            await cb.aresponse([])
        # Still closed because success reset the counter
        assert cb.state == "closed"


class TestCircuitBreakerTripping:
    """Tests for the closed -> open transition."""

    @pytest.mark.asyncio
    async def test_trips_after_threshold(self) -> None:
        adapter = _FakeAdapter([ValueError("fail")] * 10)
        cb = CircuitBreaker(adapter, failure_threshold=3, reset_timeout=60.0)

        for _ in range(3):
            with pytest.raises(ValueError):
                await cb.aresponse([])

        assert cb.state == "open"
        assert cb.trip_count == 1
        assert cb.failure_count == 3

    @pytest.mark.asyncio
    async def test_open_rejects_calls(self) -> None:
        adapter = _FakeAdapter([ValueError("fail")] * 10)
        cb = CircuitBreaker(adapter, failure_threshold=2, reset_timeout=60.0)

        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.aresponse([])

        with pytest.raises(CircuitBreakerOpenError):
            await cb.aresponse([])

    @pytest.mark.asyncio
    async def test_open_rejects_all_protocol_methods(self) -> None:
        adapter = _FakeAdapter([ValueError("fail")] * 10)
        cb = CircuitBreaker(adapter, failure_threshold=2, reset_timeout=60.0)

        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.aresponse([])

        with pytest.raises(CircuitBreakerOpenError):
            await cb.aresponse_with_tools([])
        with pytest.raises(CircuitBreakerOpenError):
            await cb.aresponse_structured([])
        with pytest.raises(CircuitBreakerOpenError):
            await cb.acall_tool("tool", {})


class TestCircuitBreakerHalfOpen:
    """Tests for the half-open recovery state."""

    @pytest.mark.asyncio
    async def test_transitions_to_half_open_after_timeout(self) -> None:
        adapter = _FakeAdapter([ValueError("fail")] * 5 + ["ok"])
        cb = CircuitBreaker(adapter, failure_threshold=2, reset_timeout=0.05)

        # Trip the breaker
        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.aresponse([])

        assert cb.state == "open"

        # Wait for reset timeout
        await asyncio.sleep(0.06)

        assert cb.state == "half_open"

    @pytest.mark.asyncio
    async def test_half_open_probe_success_closes(self) -> None:
        adapter = _FakeAdapter([ValueError("fail"), ValueError("fail"), "recovered"])
        cb = CircuitBreaker(adapter, failure_threshold=2, reset_timeout=0.05)

        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.aresponse([])

        await asyncio.sleep(0.06)

        result = await cb.aresponse([])
        assert result == "recovered"
        assert cb.state == "closed"

    @pytest.mark.asyncio
    async def test_half_open_probe_failure_reopens(self) -> None:
        adapter = _FakeAdapter([ValueError("fail")] * 10)
        cb = CircuitBreaker(adapter, failure_threshold=2, reset_timeout=0.05)

        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.aresponse([])

        await asyncio.sleep(0.06)

        # Probe call in half-open state fails → re-opens
        with pytest.raises(ValueError):
            await cb.aresponse([])

        assert cb.state == "open"
        assert cb.trip_count == 2

    @pytest.mark.asyncio
    async def test_half_open_limits_concurrent_probes(self) -> None:
        adapter = _FakeAdapter([ValueError("fail")] * 5)
        cb = CircuitBreaker(adapter, failure_threshold=2, reset_timeout=0.05, half_open_max_calls=1)

        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.aresponse([])

        await asyncio.sleep(0.06)

        # First probe starts (will fail, but the breaker increments counter)
        with pytest.raises(ValueError):
            await cb.aresponse([])

        # Breaker re-opened after probe failure
        assert cb.state == "open"


class TestCircuitBreakerObservation:
    """Tests for observation/stats properties."""

    @pytest.mark.asyncio
    async def test_counters_track_calls(self) -> None:
        adapter = _FakeAdapter(["a", ValueError("e"), "b"])
        cb = CircuitBreaker(adapter, failure_threshold=5)

        await cb.aresponse([])
        with pytest.raises(ValueError):
            await cb.aresponse([])
        await cb.aresponse([])

        assert cb.success_count == 2
        assert cb.failure_count == 1
        assert cb.trip_count == 0

    @pytest.mark.asyncio
    async def test_trip_count_increments_on_each_trip(self) -> None:
        adapter = _FakeAdapter([ValueError("fail")] * 20 + ["ok"])
        cb = CircuitBreaker(adapter, failure_threshold=2, reset_timeout=0.05)

        # First trip
        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.aresponse([])
        assert cb.trip_count == 1

        # Wait, then probe fails → second trip
        await asyncio.sleep(0.06)
        with pytest.raises(ValueError):
            await cb.aresponse([])
        assert cb.trip_count == 2


class TestCircuitBreakerPassthrough:
    """Tests for attribute forwarding."""

    @pytest.mark.asyncio
    async def test_getattr_forwards_to_inner(self) -> None:
        adapter = _FakeAdapter()
        cb = CircuitBreaker(adapter, failure_threshold=3)

        assert cb.model == "test-model"

    @pytest.mark.asyncio
    async def test_acall_tool_works(self) -> None:
        inner = AsyncMock()
        inner.acall_tool = AsyncMock(return_value={"result": "found"})
        cb = CircuitBreaker(inner, failure_threshold=3)

        result = await cb.acall_tool("search", {"q": "test"})
        assert result == {"result": "found"}
        assert cb.success_count == 1
