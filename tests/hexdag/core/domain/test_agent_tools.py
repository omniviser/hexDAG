"""Tests for agent lifecycle tools and ToolRouter integration."""

from __future__ import annotations

from typing import Any

import pytest

from hexdag.core.domain.agent_tools import CHANGE_PHASE, TOOL_END, change_phase, tool_end
from hexdag.core.exceptions import ResourceNotFoundError
from hexdag.core.ports.tool_router import ToolRouter

# ── Helpers ──────────────────────────────────────────────────────────


def _make_lifecycle_router(
    on_phase_change: Any = None,
) -> ToolRouter:
    """Create a ToolRouter with agent lifecycle tools, mimicking agent_node.py."""
    if on_phase_change is not None:

        def _change_phase_with_callback(**kw: Any) -> dict[str, Any]:
            result = change_phase(**kw)
            on_phase_change(result)
            return result

        tools = {TOOL_END: tool_end, CHANGE_PHASE: _change_phase_with_callback}
    else:
        tools = {TOOL_END: tool_end, CHANGE_PHASE: change_phase}
    return ToolRouter(tools=tools)


# ── Constants ────────────────────────────────────────────────────────


class TestConstants:
    """Verify exported constants."""

    def test_tool_end_constant(self) -> None:
        assert TOOL_END == "tool_end"

    def test_change_phase_constant(self) -> None:
        assert CHANGE_PHASE == "change_phase"


# ── Plain function tests ─────────────────────────────────────────────


class TestToolEndFunction:
    """Tests for tool_end plain function."""

    def test_returns_kwargs(self) -> None:
        assert tool_end(a=1, b="hello") == {"a": 1, "b": "hello"}

    def test_empty_kwargs(self) -> None:
        assert tool_end() == {}

    def test_nested_data(self) -> None:
        result = tool_end(data={"nested": [1, 2, 3]})
        assert result == {"data": {"nested": [1, 2, 3]}}


class TestChangePhaseFunction:
    """Tests for change_phase plain function."""

    def test_basic(self) -> None:
        result = change_phase(phase="analysis", reason="done gathering")
        assert result["action"] == "change_phase"
        assert result["new_phase"] == "analysis"
        assert result["context"]["reason"] == "done gathering"

    def test_with_carried_data(self) -> None:
        result = change_phase(phase="synthesis", carried_data={"findings": [1, 2, 3]})
        assert result["context"]["carried_data"] == {"findings": [1, 2, 3]}

    def test_all_fields(self) -> None:
        result = change_phase(
            phase="final",
            previous_phase="analysis",
            reason="done",
            carried_data={"key": "val"},
            target_output="summary",
            iteration=3,
            metadata={"source": "test"},
        )
        ctx = result["context"]
        assert ctx["previous_phase"] == "analysis"
        assert ctx["reason"] == "done"
        assert ctx["carried_data"] == {"key": "val"}
        assert ctx["target_output"] == "summary"
        assert ctx["iteration"] == 3
        assert ctx["metadata"] == {"source": "test"}

    def test_none_fields_omitted(self) -> None:
        result = change_phase(phase="test")
        assert result["context"] == {}


# ── ToolRouter with lifecycle tools ──────────────────────────────────


class TestToolRouterLifecycleTools:
    """Tests for ToolRouter loaded with lifecycle tools."""

    @pytest.mark.asyncio
    async def test_tool_end_returns_kwargs(self) -> None:
        router = _make_lifecycle_router()
        result = await router.acall_tool("tool_end", {"a": 1, "b": "hello"})
        assert result == {"a": 1, "b": "hello"}

    @pytest.mark.asyncio
    async def test_tool_end_empty_kwargs(self) -> None:
        router = _make_lifecycle_router()
        result = await router.acall_tool("tool_end", {})
        assert result == {}

    @pytest.mark.asyncio
    async def test_change_phase_basic(self) -> None:
        router = _make_lifecycle_router()
        result = await router.acall_tool(
            "change_phase", {"phase": "analysis", "reason": "done gathering"}
        )
        assert result["action"] == "change_phase"
        assert result["new_phase"] == "analysis"
        assert result["context"]["reason"] == "done gathering"

    @pytest.mark.asyncio
    async def test_change_phase_fires_callback(self) -> None:
        captured: list[dict[str, Any]] = []
        router = _make_lifecycle_router(on_phase_change=captured.append)
        await router.acall_tool("change_phase", {"phase": "review"})
        assert len(captured) == 1
        assert captured[0]["new_phase"] == "review"

    @pytest.mark.asyncio
    async def test_tool_end_does_not_fire_callback(self) -> None:
        captured: list[dict[str, Any]] = []
        router = _make_lifecycle_router(on_phase_change=captured.append)
        await router.acall_tool("tool_end", {"output": "done"})
        assert len(captured) == 0

    @pytest.mark.asyncio
    async def test_unknown_tool_raises(self) -> None:
        router = _make_lifecycle_router()
        with pytest.raises(ResourceNotFoundError) as exc_info:
            await router.acall_tool("nonexistent", {})
        assert exc_info.value.resource_type == "tool"
        assert exc_info.value.resource_id == "nonexistent"

    @pytest.mark.asyncio
    async def test_change_phase_filters_extra_params(self) -> None:
        """Extra params not in change_phase signature are ignored."""
        router = _make_lifecycle_router()
        result = await router.acall_tool(
            "change_phase",
            {"phase": "test", "unknown_param": "ignored"},
        )
        assert result["new_phase"] == "test"

    @pytest.mark.asyncio
    async def test_change_phase_all_fields(self) -> None:
        router = _make_lifecycle_router()
        result = await router.acall_tool(
            "change_phase",
            {
                "phase": "final",
                "previous_phase": "analysis",
                "reason": "done",
                "carried_data": {"key": "val"},
                "target_output": "summary",
                "iteration": 3,
                "metadata": {"source": "test"},
            },
        )
        ctx = result["context"]
        assert ctx["previous_phase"] == "analysis"
        assert ctx["reason"] == "done"
        assert ctx["carried_data"] == {"key": "val"}
        assert ctx["target_output"] == "summary"
        assert ctx["iteration"] == 3
        assert ctx["metadata"] == {"source": "test"}


# ── Schema methods ────────────────────────────────────────────────────


class TestSchemas:
    """Tests for schema discovery methods."""

    def test_get_available_tools(self) -> None:
        router = _make_lifecycle_router()
        tools = router.get_available_tools()
        assert "tool_end" in tools
        assert "change_phase" in tools

    def test_get_tool_schema_tool_end(self) -> None:
        router = _make_lifecycle_router()
        schema = router.get_tool_schema("tool_end")
        assert "description" in schema
        assert "parameters" in schema

    def test_get_tool_schema_change_phase(self) -> None:
        router = _make_lifecycle_router()
        schema = router.get_tool_schema("change_phase")
        assert "description" in schema
        param_names = [p["name"] for p in schema["parameters"]]
        assert "phase" in param_names

    def test_get_tool_schema_unknown_returns_empty(self) -> None:
        router = _make_lifecycle_router()
        schema = router.get_tool_schema("nonexistent")
        assert schema == {}

    def test_get_all_tool_schemas(self) -> None:
        router = _make_lifecycle_router()
        schemas = router.get_all_tool_schemas()
        assert "tool_end" in schemas
        assert "change_phase" in schemas
        assert len(schemas) == 2


# ── ToolRouter concrete class ────────────────────────────────────────


class TestToolRouterConcrete:
    """Tests for ToolRouter as a concrete class."""

    def test_isinstance_check(self) -> None:
        router = _make_lifecycle_router()
        assert isinstance(router, ToolRouter)

    @pytest.mark.asyncio
    async def test_aget_available_tools(self) -> None:
        router = _make_lifecycle_router()
        tools = await router.aget_available_tools()
        assert "tool_end" in tools
        assert "change_phase" in tools

    @pytest.mark.asyncio
    async def test_aget_tool_schema(self) -> None:
        router = _make_lifecycle_router()
        schema = await router.aget_tool_schema("change_phase")
        assert "description" in schema

    @pytest.mark.asyncio
    async def test_call_history_tracking(self) -> None:
        router = _make_lifecycle_router()
        await router.acall_tool("tool_end", {"x": 1})
        assert len(router.call_history) == 1
        assert router.call_history[0]["tool_name"] == "tool_end"

    def test_add_tool(self) -> None:
        router = ToolRouter()
        router.add_tool("greet", lambda name: f"Hello {name}")
        assert "greet" in router.get_available_tools()

    def test_add_tools_from(self) -> None:
        router1 = ToolRouter(tools={"a": lambda: 1})
        router2 = ToolRouter(tools={"b": lambda: 2})
        router1.add_tools_from(router2)
        assert "a" in router1.get_available_tools()
        assert "b" in router1.get_available_tools()

    def test_add_tools_from_no_overwrite(self) -> None:
        router1 = ToolRouter(tools={"a": lambda: 1})
        router2 = ToolRouter(tools={"a": lambda: 999})
        router1.add_tools_from(router2)
        # Original should not be overwritten
        assert router1._tools["a"]() == 1

    @pytest.mark.asyncio
    async def test_health_check(self) -> None:
        router = _make_lifecycle_router()
        health = await router.ahealth_check()
        assert health["status"] == "healthy"
        assert health["tool_count"] == 2

    @pytest.mark.asyncio
    async def test_async_tool(self) -> None:
        async def async_fn(x: int) -> int:
            return x * 2

        router = ToolRouter(tools={"double": async_fn})
        result = await router.acall_tool("double", {"x": 5})
        assert result == 10

    def test_non_callable_raises(self) -> None:
        with pytest.raises(TypeError, match="not callable"):
            ToolRouter(tools={"bad": 42})
