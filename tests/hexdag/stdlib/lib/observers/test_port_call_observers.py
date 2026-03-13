"""Tests for PortCallStoreObserver and PortCallLogObserver."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from hexdag.kernel.orchestration.events.events import NodeCompleted
from hexdag.kernel.ports.llm import LLMPortCall
from hexdag.kernel.ports.tool_router import ToolRouterPortCall
from hexdag.stdlib.lib.observers.port_call_observers import (
    PortCallLogObserver,
    PortCallStoreObserver,
    StoredPortCall,
)

# ---------------------------------------------------------------------------
# Fixtures — sample port call events
# ---------------------------------------------------------------------------


def _llm_event(
    *,
    node_name: str = "analyzer",
    method: str = "aresponse",
    duration_ms: float = 150.0,
    model: str = "gpt-4o",
    usage: dict[str, int] | None = None,
    response: str = "Hello world",
) -> LLMPortCall:
    return LLMPortCall(
        port_type="llm",
        method=method,
        node_name=node_name,
        duration_ms=duration_ms,
        model=model,
        usage=usage or {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        response=response,
    )


def _tool_event(
    *,
    node_name: str = "agent",
    tool_name: str = "search_db",
    params: dict[str, Any] | None = None,
    result: Any = None,
    duration_ms: float = 25.0,
) -> ToolRouterPortCall:
    return ToolRouterPortCall(
        port_type="tool_router",
        method="acall_tool",
        node_name=node_name,
        duration_ms=duration_ms,
        tool_name=tool_name,
        params=params or {"query": "test"},
        result=result or {"rows": 5},
    )


# ===========================================================================
# PortCallStoreObserver tests
# ===========================================================================


class TestPortCallStoreObserver:
    """Tests for PortCallStoreObserver."""

    @pytest.mark.asyncio
    async def test_stores_llm_port_call(self) -> None:
        """LLMPortCall events are stored with correct fields."""
        observer = PortCallStoreObserver()
        event = _llm_event()

        await observer.handle(event)

        assert observer.call_count == 1
        calls = observer.get_calls()
        assert len(calls) == 1
        assert calls[0].port_type == "llm"
        assert calls[0].method == "aresponse"
        assert calls[0].node_name == "analyzer"
        assert calls[0].duration_ms == 150.0
        assert calls[0].details["model"] == "gpt-4o"
        assert calls[0].details["usage"]["total_tokens"] == 150

    @pytest.mark.asyncio
    async def test_stores_tool_router_port_call(self) -> None:
        """ToolRouterPortCall events are stored with tool details."""
        observer = PortCallStoreObserver()
        event = _tool_event()

        await observer.handle(event)

        assert observer.call_count == 1
        calls = observer.get_calls()
        assert calls[0].port_type == "tool_router"
        assert calls[0].details["tool_name"] == "search_db"
        assert calls[0].details["params"] == {"query": "test"}

    @pytest.mark.asyncio
    async def test_ignores_non_port_call_events(self) -> None:
        """Non-PortCallEvent events are ignored."""
        observer = PortCallStoreObserver()
        event = NodeCompleted(name="node1", wave_index=0, result={}, duration_ms=100.0)

        await observer.handle(event)

        assert observer.call_count == 0

    @pytest.mark.asyncio
    async def test_filter_by_port_type(self) -> None:
        """get_calls filters by port_type."""
        observer = PortCallStoreObserver()
        await observer.handle(_llm_event())
        await observer.handle(_tool_event())
        await observer.handle(_llm_event(node_name="summarizer"))

        llm_calls = observer.get_calls(port_type="llm")
        tool_calls = observer.get_calls(port_type="tool_router")

        assert len(llm_calls) == 2
        assert len(tool_calls) == 1

    @pytest.mark.asyncio
    async def test_filter_by_node_name(self) -> None:
        """get_calls filters by node_name."""
        observer = PortCallStoreObserver()
        await observer.handle(_llm_event(node_name="analyzer"))
        await observer.handle(_llm_event(node_name="summarizer"))
        await observer.handle(_tool_event(node_name="analyzer"))

        calls = observer.get_calls(node_name="analyzer")
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_filter_by_method(self) -> None:
        """get_calls filters by method."""
        observer = PortCallStoreObserver()
        await observer.handle(_llm_event(method="aresponse"))
        await observer.handle(_llm_event(method="aresponse_with_tools"))
        await observer.handle(_tool_event())

        calls = observer.get_calls(method="aresponse")
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_combined_filters(self) -> None:
        """Multiple filters are combined (AND logic)."""
        observer = PortCallStoreObserver()
        await observer.handle(_llm_event(node_name="a", method="aresponse"))
        await observer.handle(_llm_event(node_name="a", method="aresponse_with_tools"))
        await observer.handle(_llm_event(node_name="b", method="aresponse"))

        calls = observer.get_calls(node_name="a", method="aresponse")
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_get_summary(self) -> None:
        """get_summary aggregates port call data."""
        observer = PortCallStoreObserver()
        await observer.handle(_llm_event(duration_ms=100.0))
        await observer.handle(_llm_event(duration_ms=200.0, node_name="summarizer"))
        await observer.handle(_tool_event(duration_ms=50.0))

        summary = observer.get_summary()
        assert summary["total_calls"] == 3
        assert summary["by_port_type"] == {"llm": 2, "tool_router": 1}
        assert summary["by_node"] == {"analyzer": 1, "summarizer": 1, "agent": 1}
        assert summary["total_duration_ms"] == 350.0

    @pytest.mark.asyncio
    async def test_reset(self) -> None:
        """reset() clears all stored calls."""
        observer = PortCallStoreObserver()
        await observer.handle(_llm_event())
        assert observer.call_count == 1

        observer.reset()
        assert observer.call_count == 0
        assert observer.get_calls() == []

    @pytest.mark.asyncio
    async def test_save_requires_storage(self) -> None:
        """save() raises RuntimeError without storage."""
        observer = PortCallStoreObserver()
        await observer.handle(_llm_event())

        with pytest.raises(RuntimeError, match="No storage configured"):
            await observer.save("run-1")

    @pytest.mark.asyncio
    async def test_save_and_load(self) -> None:
        """save() persists to storage, load() retrieves it."""
        mock_storage = AsyncMock()
        mock_storage.aset = AsyncMock()
        mock_storage.aget = AsyncMock(return_value=[{"port_type": "llm"}])

        observer = PortCallStoreObserver(storage=mock_storage)
        await observer.handle(_llm_event())

        count = await observer.save("run-1")
        assert count == 1
        mock_storage.aset.assert_called_once()
        key = mock_storage.aset.call_args[0][0]
        assert key == "port_calls:run-1"

        result = await observer.load("run-1")
        assert result == [{"port_type": "llm"}]

    @pytest.mark.asyncio
    async def test_load_without_storage(self) -> None:
        """load() returns None without storage."""
        observer = PortCallStoreObserver()
        result = await observer.load("run-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_llm_details_omit_full_response(self) -> None:
        """LLM details include response_length, not full response text."""
        observer = PortCallStoreObserver()
        await observer.handle(_llm_event(response="A" * 1000))

        details = observer.get_calls()[0].details
        assert "response" not in details
        assert details["response_length"] == 1000

    @pytest.mark.asyncio
    async def test_tool_error_captured(self) -> None:
        """Tool call errors are captured in details."""
        observer = PortCallStoreObserver()
        await observer.handle(_tool_event(result={"error": "permission denied"}))

        details = observer.get_calls()[0].details
        assert details["error"] == "permission denied"


# ===========================================================================
# PortCallLogObserver tests
# ===========================================================================


class TestPortCallLogObserver:
    """Tests for PortCallLogObserver."""

    @pytest.mark.asyncio
    async def test_logs_llm_port_call(self) -> None:
        """LLMPortCall is logged as structured JSON."""
        observer = PortCallLogObserver()
        event = _llm_event()

        # Patch the logger to capture output
        logged: list[str] = []
        observer._logger = type(
            "MockLogger",
            (),
            {
                "info": lambda self, fmt, *args: logged.append(fmt.format(*args)),
            },
        )()

        await observer.handle(event)

        assert observer.call_count == 1
        assert len(logged) == 1
        parsed = json.loads(logged[0])
        assert parsed["event"] == "port_call"
        assert parsed["port_type"] == "llm"
        assert parsed["method"] == "aresponse"
        assert parsed["node_name"] == "analyzer"
        assert parsed["details"]["model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_logs_tool_router_port_call(self) -> None:
        """ToolRouterPortCall is logged with tool details."""
        observer = PortCallLogObserver()
        logged: list[str] = []
        observer._logger = type(
            "MockLogger",
            (),
            {
                "info": lambda self, fmt, *args: logged.append(fmt.format(*args)),
            },
        )()

        await observer.handle(_tool_event())

        parsed = json.loads(logged[0])
        assert parsed["port_type"] == "tool_router"
        assert parsed["details"]["tool_name"] == "search_db"

    @pytest.mark.asyncio
    async def test_ignores_non_port_call_events(self) -> None:
        """Non-PortCallEvent events are ignored."""
        observer = PortCallLogObserver()
        event = NodeCompleted(name="node1", wave_index=0, result={}, duration_ms=100.0)

        await observer.handle(event)
        assert observer.call_count == 0

    @pytest.mark.asyncio
    async def test_include_details_false(self) -> None:
        """With include_details=False, details are omitted."""
        observer = PortCallLogObserver(include_details=False)
        logged: list[str] = []
        observer._logger = type(
            "MockLogger",
            (),
            {
                "info": lambda self, fmt, *args: logged.append(fmt.format(*args)),
            },
        )()

        await observer.handle(_llm_event())

        parsed = json.loads(logged[0])
        assert "details" not in parsed
        assert parsed["port_type"] == "llm"

    @pytest.mark.asyncio
    async def test_call_count_increments(self) -> None:
        """call_count tracks total logged events."""
        observer = PortCallLogObserver()
        logged: list[str] = []
        observer._logger = type(
            "MockLogger",
            (),
            {
                "info": lambda self, fmt, *args: logged.append(fmt.format(*args)),
            },
        )()

        await observer.handle(_llm_event())
        await observer.handle(_tool_event())
        await observer.handle(_llm_event())

        assert observer.call_count == 3

    @pytest.mark.asyncio
    async def test_reset(self) -> None:
        """reset() clears the call counter."""
        observer = PortCallLogObserver()
        logged: list[str] = []
        observer._logger = type(
            "MockLogger",
            (),
            {
                "info": lambda self, fmt, *args: logged.append(fmt.format(*args)),
            },
        )()

        await observer.handle(_llm_event())
        assert observer.call_count == 1

        observer.reset()
        assert observer.call_count == 0


# ===========================================================================
# StoredPortCall tests
# ===========================================================================


class TestStoredPortCall:
    """Tests for StoredPortCall dataclass."""

    def test_default_details(self) -> None:
        """Default details is empty dict."""
        record = StoredPortCall(
            port_type="llm",
            method="aresponse",
            node_name="test",
            duration_ms=100.0,
            timestamp=1234567890.0,
        )
        assert record.details == {}

    def test_with_details(self) -> None:
        """StoredPortCall stores arbitrary details."""
        record = StoredPortCall(
            port_type="llm",
            method="aresponse",
            node_name="test",
            duration_ms=100.0,
            timestamp=1234567890.0,
            details={"model": "gpt-4o", "usage": {"total_tokens": 150}},
        )
        assert record.details["model"] == "gpt-4o"
