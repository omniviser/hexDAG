"""Tests for ToolCallNode -- execute a single tool call as a node."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from hexdag.kernel.domain.dag import NodeSpec
from hexdag.stdlib.nodes.tool_call_node import ToolCallNode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sync_tool(query: str, limit: int = 10) -> list[str]:
    """A synchronous test tool."""
    return [f"result_{i}" for i in range(min(limit, 3))]


async def async_tool(query: str) -> dict[str, str]:
    """An asynchronous test tool."""
    return {"answer": f"found: {query}"}


# ---------------------------------------------------------------------------
# TestToolCallNodeCreation
# ---------------------------------------------------------------------------


class TestToolCallNodeCreation:
    """Tests for ToolCallNode factory __call__."""

    def setup_method(self) -> None:
        self.factory = ToolCallNode()

    def test_creates_node_spec(self) -> None:
        """Factory creates a valid NodeSpec."""
        spec = self.factory(
            name="search_tool",
            tool_name="myapp.tools.search",
            arguments={"query": "test"},
        )
        assert isinstance(spec, NodeSpec)
        assert spec.name == "search_tool"
        assert spec.fn is not None

    def test_includes_dependencies(self) -> None:
        """Dependencies are passed through."""
        spec = self.factory(
            name="tool_node",
            tool_name="myapp.tools.search",
            deps=["llm_node"],
        )
        assert "llm_node" in spec.deps

    def test_default_empty_arguments(self) -> None:
        """Arguments default to empty dict."""
        spec = self.factory(
            name="tool_node",
            tool_name="myapp.tools.no_args",
        )
        assert spec.fn is not None

    def test_with_tool_call_id(self) -> None:
        """tool_call_id is accepted without error."""
        spec = self.factory(
            name="tool_node",
            tool_name="myapp.tools.search",
            tool_call_id="call_abc123",
        )
        assert spec.fn is not None


# ---------------------------------------------------------------------------
# TestToolCallNodeExecution
# ---------------------------------------------------------------------------


class TestToolCallNodeExecution:
    """Tests for ToolCallNode execution."""

    def setup_method(self) -> None:
        self.factory = ToolCallNode()

    @pytest.mark.asyncio
    async def test_executes_sync_tool(self) -> None:
        """Sync tool is resolved and called."""
        spec = self.factory(
            name="sync_tool",
            tool_name="tests.hexdag.stdlib.nodes.test_tool_call_node.sync_tool",
            arguments={"query": "test", "limit": 2},
        )

        with (
            patch(
                "hexdag.stdlib.nodes.tool_call_node.resolve_function",
                return_value=sync_tool,
            ),
            patch(
                "hexdag.stdlib.nodes.tool_call_node.get_port",
                return_value=None,
            ),
        ):
            result = await spec.fn({})

        assert result == ["result_0", "result_1"]

    @pytest.mark.asyncio
    async def test_executes_async_tool(self) -> None:
        """Async tool is resolved and awaited."""
        spec = self.factory(
            name="async_tool",
            tool_name="tests.hexdag.stdlib.nodes.test_tool_call_node.async_tool",
            arguments={"query": "python"},
        )

        with (
            patch(
                "hexdag.stdlib.nodes.tool_call_node.resolve_function",
                return_value=async_tool,
            ),
            patch(
                "hexdag.stdlib.nodes.tool_call_node.get_port",
                return_value=None,
            ),
        ):
            result = await spec.fn({})

        assert result == {"answer": "found: python"}

    @pytest.mark.asyncio
    async def test_emits_tool_router_event(self) -> None:
        """ToolRouterEvent is emitted when observer_manager is available."""
        observer = AsyncMock()
        observer.notify = AsyncMock()

        spec = self.factory(
            name="event_tool",
            tool_name="myapp.tool",
            arguments={"x": 1},
        )

        with (
            patch(
                "hexdag.stdlib.nodes.tool_call_node.resolve_function",
                return_value=lambda x: x * 2,
            ),
            patch(
                "hexdag.stdlib.nodes.tool_call_node.get_port",
                return_value=observer,
            ),
        ):
            result = await spec.fn({})

        assert result == 2
        observer.notify.assert_called_once()

    @pytest.mark.asyncio
    async def test_raises_on_tool_failure(self) -> None:
        """Tool execution errors propagate."""

        def bad_tool(**kwargs: Any) -> None:
            raise RuntimeError("tool broke")

        spec = self.factory(
            name="bad_tool",
            tool_name="myapp.bad",
            arguments={},
        )

        with (
            patch(
                "hexdag.stdlib.nodes.tool_call_node.resolve_function",
                return_value=bad_tool,
            ),
            patch(
                "hexdag.stdlib.nodes.tool_call_node.get_port",
                return_value=None,
            ),
            pytest.raises(RuntimeError, match="tool broke"),
        ):
            await spec.fn({})

    @pytest.mark.asyncio
    async def test_no_observer_does_not_fail(self) -> None:
        """Execution succeeds even without observer_manager."""
        spec = self.factory(
            name="no_obs_tool",
            tool_name="myapp.tool",
            arguments={"val": "hello"},
        )

        with (
            patch(
                "hexdag.stdlib.nodes.tool_call_node.resolve_function",
                return_value=lambda val: val.upper(),
            ),
            patch(
                "hexdag.stdlib.nodes.tool_call_node.get_port",
                side_effect=Exception("no port"),
            ),
        ):
            result = await spec.fn({})

        assert result == "HELLO"
