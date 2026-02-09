"""Tests for AgentToolRouter and agent lifecycle tools."""

from __future__ import annotations

from typing import Any

import pytest

from hexdag.core.domain.agent_tools import (
    CHANGE_PHASE,
    TOOL_END,
    AgentToolRouter,
)
from hexdag.core.exceptions import ResourceNotFoundError

# ── Constants ────────────────────────────────────────────────────────


class TestConstants:
    """Verify exported constants."""

    def test_tool_end_constant(self) -> None:
        assert TOOL_END == "tool_end"

    def test_change_phase_constant(self) -> None:
        assert CHANGE_PHASE == "change_phase"


# ── AgentToolRouter init ─────────────────────────────────────────────


class TestAgentToolRouterInit:
    """Tests for AgentToolRouter initialization."""

    def test_default_init(self) -> None:
        router = AgentToolRouter()
        assert router.get_available_tools() == ["tool_end", "change_phase"]

    def test_init_with_callback(self) -> None:
        def cb(result: dict[str, Any]) -> None:
            pass

        router = AgentToolRouter(on_phase_change=cb)
        assert router._on_phase_change is cb


# ── acall_tool ────────────────────────────────────────────────────────


class TestAcallTool:
    """Tests for acall_tool method."""

    @pytest.mark.asyncio
    async def test_tool_end_returns_kwargs(self) -> None:
        router = AgentToolRouter()
        result = await router.acall_tool("tool_end", {"a": 1, "b": "hello"})
        assert result == {"a": 1, "b": "hello"}

    @pytest.mark.asyncio
    async def test_tool_end_empty_kwargs(self) -> None:
        router = AgentToolRouter()
        result = await router.acall_tool("tool_end", {})
        assert result == {}

    @pytest.mark.asyncio
    async def test_change_phase_basic(self) -> None:
        router = AgentToolRouter()
        result = await router.acall_tool(
            "change_phase", {"phase": "analysis", "reason": "done gathering"}
        )
        assert result["action"] == "change_phase"
        assert result["new_phase"] == "analysis"
        assert result["context"]["reason"] == "done gathering"

    @pytest.mark.asyncio
    async def test_change_phase_with_carried_data(self) -> None:
        router = AgentToolRouter()
        result = await router.acall_tool(
            "change_phase",
            {
                "phase": "synthesis",
                "carried_data": {"findings": [1, 2, 3]},
            },
        )
        assert result["context"]["carried_data"] == {"findings": [1, 2, 3]}

    @pytest.mark.asyncio
    async def test_change_phase_fires_callback(self) -> None:
        captured: list[dict[str, Any]] = []

        def on_phase(result: dict[str, Any]) -> None:
            captured.append(result)

        router = AgentToolRouter(on_phase_change=on_phase)
        await router.acall_tool("change_phase", {"phase": "review"})
        assert len(captured) == 1
        assert captured[0]["new_phase"] == "review"

    @pytest.mark.asyncio
    async def test_tool_end_does_not_fire_callback(self) -> None:
        captured: list[dict[str, Any]] = []

        def on_phase(result: dict[str, Any]) -> None:
            captured.append(result)

        router = AgentToolRouter(on_phase_change=on_phase)
        await router.acall_tool("tool_end", {"output": "done"})
        assert len(captured) == 0

    @pytest.mark.asyncio
    async def test_unknown_tool_raises(self) -> None:
        router = AgentToolRouter()
        with pytest.raises(ResourceNotFoundError) as exc_info:
            await router.acall_tool("nonexistent", {})
        assert exc_info.value.resource_type == "tool"
        assert exc_info.value.resource_id == "nonexistent"

    @pytest.mark.asyncio
    async def test_change_phase_filters_extra_params(self) -> None:
        """Extra params not in change_phase signature are ignored."""
        router = AgentToolRouter()
        result = await router.acall_tool(
            "change_phase",
            {"phase": "test", "unknown_param": "ignored"},
        )
        assert result["new_phase"] == "test"

    @pytest.mark.asyncio
    async def test_change_phase_all_fields(self) -> None:
        router = AgentToolRouter()
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
        router = AgentToolRouter()
        tools = router.get_available_tools()
        assert "tool_end" in tools
        assert "change_phase" in tools

    def test_get_tool_schema_tool_end(self) -> None:
        router = AgentToolRouter()
        schema = router.get_tool_schema("tool_end")
        assert "description" in schema
        assert "parameters" in schema

    def test_get_tool_schema_change_phase(self) -> None:
        router = AgentToolRouter()
        schema = router.get_tool_schema("change_phase")
        assert "description" in schema
        param_names = [p["name"] for p in schema["parameters"]]
        assert "phase" in param_names

    def test_get_tool_schema_unknown_returns_empty(self) -> None:
        router = AgentToolRouter()
        schema = router.get_tool_schema("nonexistent")
        assert schema == {}

    def test_get_all_tool_schemas(self) -> None:
        router = AgentToolRouter()
        schemas = router.get_all_tool_schemas()
        assert "tool_end" in schemas
        assert "change_phase" in schemas
        assert len(schemas) == 2


# ── ToolRouter protocol compliance ────────────────────────────────────


class TestToolRouterProtocol:
    """Verify AgentToolRouter satisfies ToolRouter protocol."""

    def test_isinstance_check(self) -> None:
        from hexdag.core.ports.tool_router import ToolRouter

        router = AgentToolRouter()
        assert isinstance(router, ToolRouter)

    @pytest.mark.asyncio
    async def test_aget_available_tools(self) -> None:
        router = AgentToolRouter()
        tools = await router.aget_available_tools()
        assert "tool_end" in tools
        assert "change_phase" in tools

    @pytest.mark.asyncio
    async def test_aget_tool_schema(self) -> None:
        router = AgentToolRouter()
        schema = await router.aget_tool_schema("change_phase")
        assert "description" in schema
