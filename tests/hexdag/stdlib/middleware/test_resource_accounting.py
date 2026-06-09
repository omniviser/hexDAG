"""Tests for ResourceAccounting domain models and observer."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from hexdag.kernel.domain.resource_accounting import (
    ResourceLimits,
    ResourceUsage,
)
from hexdag.kernel.orchestration.events.events import (
    PortCallEvent,
    ResourceLimitExceeded,
    ResourceWarning,
)
from hexdag.kernel.ports.llm import LLMPortCall
from hexdag.kernel.tool_router import ToolRouterCall
from hexdag.stdlib.middleware.resource_accounting import (
    ResourceAccountingObserver,
    ResourceLimitExceededError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _llm_event(
    input_tokens: int = 100,
    output_tokens: int = 50,
    duration_ms: float = 150.0,
    node_name: str = "analyzer",
) -> LLMPortCall:
    return LLMPortCall(
        port_type="llm",
        method="aresponse",
        node_name=node_name,
        duration_ms=duration_ms,
        usage={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        model="gpt-4o",
        response="Hello",
    )


def _tool_event(
    duration_ms: float = 50.0,
    node_name: str = "agent",
) -> ToolRouterCall:
    return ToolRouterCall(
        port_type="tool_router",
        method="acall_tool",
        node_name=node_name,
        duration_ms=duration_ms,
        tool_name="search",
    )


def _generic_port_event(duration_ms: float = 30.0) -> PortCallEvent:
    return PortCallEvent(
        port_type="data_store",
        method="aget",
        node_name="loader",
        duration_ms=duration_ms,
    )


# ---------------------------------------------------------------------------
# ResourceUsage model tests
# ---------------------------------------------------------------------------


class TestResourceUsage:
    def test_add_llm_call(self) -> None:
        usage = ResourceUsage()
        usage.add_llm_call(input_tokens=100, output_tokens=50, duration_ms=200.0)

        assert usage.total_tokens == 150
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.llm_calls == 1
        assert usage.total_duration_ms == 200.0

    def test_add_multiple_llm_calls(self) -> None:
        usage = ResourceUsage()
        usage.add_llm_call(input_tokens=100, output_tokens=50, duration_ms=200.0)
        usage.add_llm_call(input_tokens=200, output_tokens=100, duration_ms=300.0)

        assert usage.total_tokens == 450
        assert usage.llm_calls == 2
        assert usage.total_duration_ms == 500.0

    def test_add_tool_call(self) -> None:
        usage = ResourceUsage()
        usage.add_tool_call(duration_ms=50.0)

        assert usage.tool_calls == 1
        assert usage.total_duration_ms == 50.0


# ---------------------------------------------------------------------------
# ResourceLimits model tests
# ---------------------------------------------------------------------------


class TestResourceLimits:
    def test_check_all_ok(self) -> None:
        limits = ResourceLimits(max_total_tokens=1000, max_llm_calls=10)
        usage = ResourceUsage(total_tokens=100, llm_calls=2)

        checks = limits.check(usage)
        assert len(checks) == 2
        assert all(c.status == "ok" for c in checks)

    def test_check_warning(self) -> None:
        limits = ResourceLimits(max_total_tokens=1000, warning_threshold=0.8)
        usage = ResourceUsage(total_tokens=850)

        checks = limits.check(usage)
        assert len(checks) == 1
        assert checks[0].status == "warning"
        assert checks[0].resource == "total_tokens"

    def test_check_exceeded(self) -> None:
        limits = ResourceLimits(max_llm_calls=5)
        usage = ResourceUsage(llm_calls=6)

        checks = limits.check(usage)
        assert len(checks) == 1
        assert checks[0].status == "exceeded"

    def test_check_no_limits_configured(self) -> None:
        limits = ResourceLimits()  # all None
        usage = ResourceUsage(total_tokens=999999)

        checks = limits.check(usage)
        assert len(checks) == 0

    def test_check_duration(self) -> None:
        limits = ResourceLimits(max_duration_ms=5000.0)
        usage = ResourceUsage(total_duration_ms=5100.0)

        checks = limits.check(usage)
        assert checks[0].status == "exceeded"
        assert checks[0].resource == "duration_ms"


# ---------------------------------------------------------------------------
# ResourceAccountingObserver tests
# ---------------------------------------------------------------------------


class TestResourceAccountingObserver:
    @pytest.mark.asyncio
    async def test_tracks_llm_usage(self) -> None:
        observer = ResourceAccountingObserver(
            limits=ResourceLimits(max_total_tokens=10000),
            pipeline_name="test",
        )

        await observer.handle(_llm_event(input_tokens=100, output_tokens=50))

        assert observer.usage.total_tokens == 150
        assert observer.usage.llm_calls == 1

    @pytest.mark.asyncio
    async def test_tracks_tool_usage(self) -> None:
        observer = ResourceAccountingObserver(
            limits=ResourceLimits(max_tool_calls=10),
            pipeline_name="test",
        )

        await observer.handle(_tool_event())

        assert observer.usage.tool_calls == 1

    @pytest.mark.asyncio
    async def test_tracks_generic_port_call_duration(self) -> None:
        observer = ResourceAccountingObserver(
            limits=ResourceLimits(max_duration_ms=5000.0),
            pipeline_name="test",
        )

        await observer.handle(_generic_port_event(duration_ms=30.0))

        assert observer.usage.total_duration_ms == 30.0

    @pytest.mark.asyncio
    async def test_ignores_non_port_call_events(self) -> None:
        from hexdag.kernel.orchestration.events import NodeStarted

        observer = ResourceAccountingObserver(
            limits=ResourceLimits(max_total_tokens=100),
            pipeline_name="test",
        )

        await observer.handle(NodeStarted(name="n", wave_index=0))

        assert observer.usage.total_tokens == 0

    @pytest.mark.asyncio
    async def test_emits_warning_event(self) -> None:
        mgr = AsyncMock()
        observer = ResourceAccountingObserver(
            limits=ResourceLimits(max_total_tokens=1000, warning_threshold=0.8),
            pipeline_name="test",
            observer_manager=mgr,
        )

        # 850 tokens = 85% of 1000 → warning
        await observer.handle(_llm_event(input_tokens=500, output_tokens=350))

        mgr.notify.assert_called_once()
        event = mgr.notify.call_args[0][0]
        assert isinstance(event, ResourceWarning)
        assert event.resource == "total_tokens"
        assert event.pipeline_name == "test"

    @pytest.mark.asyncio
    async def test_warning_only_emitted_once_per_resource(self) -> None:
        mgr = AsyncMock()
        observer = ResourceAccountingObserver(
            limits=ResourceLimits(max_total_tokens=1000, warning_threshold=0.8),
            pipeline_name="test",
            observer_manager=mgr,
        )

        # Two calls that both cross warning threshold
        await observer.handle(_llm_event(input_tokens=500, output_tokens=350))
        await observer.handle(_llm_event(input_tokens=10, output_tokens=5))

        # Warning emitted once, then exceeded emitted once
        warning_calls = [
            c for c in mgr.notify.call_args_list if isinstance(c[0][0], ResourceWarning)
        ]
        assert len(warning_calls) == 1

    @pytest.mark.asyncio
    async def test_emits_exceeded_event(self) -> None:
        mgr = AsyncMock()
        observer = ResourceAccountingObserver(
            limits=ResourceLimits(max_llm_calls=2),
            pipeline_name="test",
            observer_manager=mgr,
        )

        await observer.handle(_llm_event())
        await observer.handle(_llm_event())

        exceeded_calls = [
            c for c in mgr.notify.call_args_list if isinstance(c[0][0], ResourceLimitExceeded)
        ]
        assert len(exceeded_calls) == 1
        assert exceeded_calls[0][0][0].resource == "llm_calls"

    @pytest.mark.asyncio
    async def test_enforce_raises_on_exceeded(self) -> None:
        observer = ResourceAccountingObserver(
            limits=ResourceLimits(max_llm_calls=2),
            pipeline_name="test",
            enforce=True,
        )

        await observer.handle(_llm_event())  # 1/2 — ok

        with pytest.raises(ResourceLimitExceededError, match="llm_calls exceeded"):
            await observer.handle(_llm_event())  # 2/2 — exceeded

    @pytest.mark.asyncio
    async def test_no_raise_without_enforce(self) -> None:
        observer = ResourceAccountingObserver(
            limits=ResourceLimits(max_llm_calls=2),
            pipeline_name="test",
            enforce=False,
        )

        await observer.handle(_llm_event())
        # Should not raise — just logs the exceeded event
        await observer.handle(_llm_event())

        assert observer.usage.llm_calls == 2

    @pytest.mark.asyncio
    async def test_reset_clears_state(self) -> None:
        observer = ResourceAccountingObserver(
            limits=ResourceLimits(max_total_tokens=1000, warning_threshold=0.8),
            pipeline_name="test",
        )

        await observer.handle(_llm_event(input_tokens=500, output_tokens=350))
        observer.reset()

        assert observer.usage.total_tokens == 0
        assert observer.usage.llm_calls == 0

    @pytest.mark.asyncio
    async def test_get_summary(self) -> None:
        observer = ResourceAccountingObserver(
            limits=ResourceLimits(max_total_tokens=1000, max_llm_calls=10),
            pipeline_name="test",
        )

        await observer.handle(_llm_event(input_tokens=100, output_tokens=50))

        summary = observer.get_summary()
        assert summary["usage"]["total_tokens"] == 150
        assert summary["usage"]["llm_calls"] == 1
        assert summary["limits"]["max_total_tokens"] == 1000
        assert len(summary["checks"]) == 2

    @pytest.mark.asyncio
    async def test_llm_event_without_usage_dict(self) -> None:
        """LLMPortCall with usage=None should not crash."""
        observer = ResourceAccountingObserver(
            limits=ResourceLimits(max_llm_calls=10),
            pipeline_name="test",
        )

        event = LLMPortCall(
            port_type="llm",
            method="aresponse",
            node_name="n",
            duration_ms=100.0,
            usage=None,
            response="hi",
        )
        await observer.handle(event)

        assert observer.usage.llm_calls == 1
        assert observer.usage.total_tokens == 0
