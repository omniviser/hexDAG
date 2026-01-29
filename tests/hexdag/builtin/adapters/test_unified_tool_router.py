"""Tests for the UnifiedToolRouter module."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from hexdag.builtin.adapters.unified_tool_router import UnifiedToolRouter
from hexdag.core.exceptions import ResourceNotFoundError
from hexdag.core.ports.tool_router import ToolRouter


class MockToolRouter(ToolRouter):
    """Mock tool router for testing."""

    def __init__(self, tools: dict[str, Any] | None = None) -> None:
        """Initialize with optional tools dict."""
        self.tools = tools or {}
        self._schemas: dict[str, dict[str, Any]] = {}

    async def acall_tool(self, tool_name: str, params: dict[str, Any]) -> Any:
        """Call a tool by name."""
        if tool_name not in self.tools:
            raise ResourceNotFoundError("tool", tool_name, list(self.tools.keys()))
        tool = self.tools[tool_name]
        if callable(tool):
            return tool(**params)
        return tool

    def get_available_tools(self) -> list[str]:
        """Get list of available tool names."""
        return list(self.tools.keys())

    def get_tool_schema(self, tool_name: str) -> dict[str, Any]:
        """Get schema for a tool."""
        return self._schemas.get(tool_name, {})

    def set_tool_schema(self, tool_name: str, schema: dict[str, Any]) -> None:
        """Set schema for a tool (helper for testing)."""
        self._schemas[tool_name] = schema


class TestUnifiedToolRouterInit:
    """Tests for UnifiedToolRouter initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization with no routers."""
        router = UnifiedToolRouter()
        assert router.routers == {}
        assert router.default_router is None

    def test_initialization_with_routers(self) -> None:
        """Test initialization with routers dict."""
        mock_router = MockToolRouter()
        router = UnifiedToolRouter(routers={"builtin": mock_router})
        assert "builtin" in router.routers
        assert router.default_router is mock_router

    def test_initialization_with_multiple_routers(self) -> None:
        """Test initialization with multiple routers."""
        router1 = MockToolRouter()
        router2 = MockToolRouter()
        router = UnifiedToolRouter(routers={"builtin": router1, "custom": router2})
        assert len(router.routers) == 2
        # Default is the first one
        assert router.default_router is router1


class TestAcallTool:
    """Tests for acall_tool method."""

    @pytest.fixture
    def router_with_tools(self) -> UnifiedToolRouter:
        """Create router with mock tools."""
        builtin_router = MockToolRouter(
            tools={
                "add": lambda x, y: x + y,
                "multiply": lambda x, y: x * y,
            }
        )
        custom_router = MockToolRouter(
            tools={
                "greet": lambda name: f"Hello, {name}!",
            }
        )
        return UnifiedToolRouter(routers={"builtin": builtin_router, "custom": custom_router})

    @pytest.mark.asyncio
    async def test_call_namespaced_tool(self, router_with_tools: UnifiedToolRouter) -> None:
        """Test calling a tool with namespace prefix."""
        result = await router_with_tools.acall_tool("builtin::add", {"x": 2, "y": 3})
        assert result == 5

    @pytest.mark.asyncio
    async def test_call_namespaced_tool_different_router(
        self, router_with_tools: UnifiedToolRouter
    ) -> None:
        """Test calling a tool from different router."""
        result = await router_with_tools.acall_tool("custom::greet", {"name": "World"})
        assert result == "Hello, World!"

    @pytest.mark.asyncio
    async def test_call_tool_without_namespace(self, router_with_tools: UnifiedToolRouter) -> None:
        """Test calling a tool without namespace uses default router."""
        result = await router_with_tools.acall_tool("multiply", {"x": 4, "y": 5})
        assert result == 20

    @pytest.mark.asyncio
    async def test_call_tool_unknown_router(self, router_with_tools: UnifiedToolRouter) -> None:
        """Test calling a tool from unknown router raises error."""
        with pytest.raises(ResourceNotFoundError) as exc_info:
            await router_with_tools.acall_tool("unknown::tool", {})
        assert exc_info.value.resource_type == "tool_router"
        assert exc_info.value.resource_id == "unknown"

    @pytest.mark.asyncio
    async def test_call_tool_no_default_router(self) -> None:
        """Test calling tool without namespace when no default router."""
        router = UnifiedToolRouter()
        with pytest.raises(ResourceNotFoundError) as exc_info:
            await router.acall_tool("some_tool", {})
        assert exc_info.value.resource_type == "tool"


class TestExecuteTool:
    """Tests for _execute_tool method."""

    @pytest.fixture
    def router(self) -> UnifiedToolRouter:
        """Create empty router for testing."""
        return UnifiedToolRouter()

    @pytest.mark.asyncio
    async def test_execute_async_function(self, router: UnifiedToolRouter) -> None:
        """Test executing an async function."""

        async def async_tool(x: int) -> int:
            return x * 2

        result = await router._execute_tool(async_tool, {"x": 5})
        assert result == 10

    @pytest.mark.asyncio
    async def test_execute_sync_function(self, router: UnifiedToolRouter) -> None:
        """Test executing a sync function."""

        def sync_tool(x: int) -> int:
            return x + 1

        result = await router._execute_tool(sync_tool, {"x": 5})
        assert result == 6

    @pytest.mark.asyncio
    async def test_execute_class_with_execute_method(self, router: UnifiedToolRouter) -> None:
        """Test executing a class with execute method."""

        class ToolClass:
            def execute(self, value: str) -> str:
                return value.upper()

        tool = ToolClass()
        result = await router._execute_tool(tool, {"value": "hello"})
        assert result == "HELLO"

    @pytest.mark.asyncio
    async def test_execute_class_with_async_execute(self, router: UnifiedToolRouter) -> None:
        """Test executing a class with async execute method."""

        class AsyncToolClass:
            async def execute(self, value: str) -> str:
                return value.lower()

        tool = AsyncToolClass()
        result = await router._execute_tool(tool, {"value": "HELLO"})
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_execute_non_executable_raises(self, router: UnifiedToolRouter) -> None:
        """Test that non-executable raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            await router._execute_tool("not callable", {})
        assert "not executable" in str(exc_info.value)


class TestCallWithParams:
    """Tests for _call_with_params method."""

    @pytest.fixture
    def router(self) -> UnifiedToolRouter:
        """Create empty router for testing."""
        return UnifiedToolRouter()

    def test_call_with_var_keyword(self, router: UnifiedToolRouter) -> None:
        """Test calling function with **kwargs."""

        def tool_with_kwargs(**kwargs: Any) -> dict[str, Any]:
            return kwargs

        result = router._call_with_params(
            tool_with_kwargs, {"a": 1, "b": 2, "extra": 3}, is_async=False
        )
        assert result == {"a": 1, "b": 2, "extra": 3}

    def test_call_with_filtered_params(self, router: UnifiedToolRouter) -> None:
        """Test that extra params are filtered out for regular functions."""

        def tool_without_kwargs(a: int, b: int) -> int:
            return a + b

        result = router._call_with_params(
            tool_without_kwargs, {"a": 1, "b": 2, "extra": 3}, is_async=False
        )
        assert result == 3

    def test_call_async_returns_coroutine(self, router: UnifiedToolRouter) -> None:
        """Test that async call returns a coroutine."""
        import asyncio

        async def async_tool(x: int) -> int:
            return x

        result = router._call_with_params(async_tool, {"x": 5}, is_async=True)
        assert asyncio.iscoroutine(result)
        # Clean up the coroutine
        asyncio.get_event_loop().run_until_complete(result)


class TestGetAvailableTools:
    """Tests for get_available_tools method."""

    def test_get_available_tools_empty(self) -> None:
        """Test getting tools from empty router."""
        router = UnifiedToolRouter()
        tools = router.get_available_tools()
        assert tools == []

    def test_get_available_tools_namespaced(self) -> None:
        """Test that tools are namespaced."""
        mock_router = MockToolRouter(tools={"tool1": lambda: None, "tool2": lambda: None})
        router = UnifiedToolRouter(routers={"ns": mock_router})
        tools = router.get_available_tools()
        assert "ns::tool1" in tools
        assert "ns::tool2" in tools

    def test_get_available_tools_multiple_routers(self) -> None:
        """Test getting tools from multiple routers."""
        router1 = MockToolRouter(tools={"a": lambda: None})
        router2 = MockToolRouter(tools={"b": lambda: None})
        router = UnifiedToolRouter(routers={"r1": router1, "r2": router2})
        tools = router.get_available_tools()
        assert "r1::a" in tools
        assert "r2::b" in tools

    def test_get_available_tools_handles_exceptions(self) -> None:
        """Test that exceptions are handled gracefully."""
        mock_router = MagicMock()
        mock_router.get_available_tools.side_effect = Exception("Error")
        router = UnifiedToolRouter(routers={"broken": mock_router})
        tools = router.get_available_tools()
        assert tools == []


class TestGetToolSchema:
    """Tests for get_tool_schema method."""

    def test_get_tool_schema_namespaced(self) -> None:
        """Test getting schema for namespaced tool."""
        mock_router = MockToolRouter()
        mock_router.set_tool_schema("tool1", {"description": "A tool"})
        router = UnifiedToolRouter(routers={"ns": mock_router})
        schema = router.get_tool_schema("ns::tool1")
        assert schema["description"] == "A tool"
        assert schema["_router"] == "ns"

    def test_get_tool_schema_default_router(self) -> None:
        """Test getting schema from default router."""
        mock_router = MockToolRouter()
        mock_router.set_tool_schema("tool1", {"description": "Default tool"})
        router = UnifiedToolRouter(routers={"default": mock_router})
        schema = router.get_tool_schema("tool1")
        assert schema["description"] == "Default tool"
        assert schema["_router"] == "default"

    def test_get_tool_schema_unknown_router(self) -> None:
        """Test getting schema from unknown router returns empty dict."""
        router = UnifiedToolRouter()
        schema = router.get_tool_schema("unknown::tool")
        assert schema == {}

    def test_get_tool_schema_no_default_router(self) -> None:
        """Test getting schema without default router returns empty dict."""
        router = UnifiedToolRouter()
        schema = router.get_tool_schema("tool")
        assert schema == {}

    def test_get_tool_schema_handles_exception(self) -> None:
        """Test that exceptions are handled gracefully."""
        mock_router = MagicMock()
        mock_router.get_tool_schema.side_effect = Exception("Error")
        router = UnifiedToolRouter(routers={"broken": mock_router})
        schema = router.get_tool_schema("broken::tool")
        assert schema == {}


class TestGetAllToolSchemas:
    """Tests for get_all_tool_schemas method."""

    def test_get_all_tool_schemas(self) -> None:
        """Test getting all tool schemas."""
        mock_router = MockToolRouter(tools={"tool1": lambda: None, "tool2": lambda: None})
        mock_router.set_tool_schema("tool1", {"description": "Tool 1"})
        mock_router.set_tool_schema("tool2", {"description": "Tool 2"})
        router = UnifiedToolRouter(routers={"ns": mock_router})
        schemas = router.get_all_tool_schemas()
        assert "ns::tool1" in schemas
        assert "ns::tool2" in schemas

    def test_get_all_tool_schemas_empty(self) -> None:
        """Test getting schemas from empty router."""
        router = UnifiedToolRouter()
        schemas = router.get_all_tool_schemas()
        assert schemas == {}


class TestAgetAvailableTools:
    """Tests for aget_available_tools async method."""

    @pytest.mark.asyncio
    async def test_aget_available_tools_uses_async_method(self) -> None:
        """Test that async method is used when available."""
        mock_router = AsyncMock()
        mock_router.aget_available_tools = AsyncMock(return_value=["async_tool"])
        router = UnifiedToolRouter(routers={"async": mock_router})
        tools = await router.aget_available_tools()
        assert "async::async_tool" in tools
        mock_router.aget_available_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_aget_available_tools_falls_back_to_sync(self) -> None:
        """Test fallback to sync method when async not available."""
        mock_router = MockToolRouter(tools={"sync_tool": lambda: None})
        router = UnifiedToolRouter(routers={"sync": mock_router})
        tools = await router.aget_available_tools()
        assert "sync::sync_tool" in tools

    @pytest.mark.asyncio
    async def test_aget_available_tools_handles_exception(self) -> None:
        """Test that exceptions are handled gracefully."""
        mock_router = MagicMock()
        mock_router.aget_available_tools = AsyncMock(side_effect=Exception("Error"))
        # Make sure hasattr returns True for aget_available_tools
        router = UnifiedToolRouter(routers={"broken": mock_router})
        tools = await router.aget_available_tools()
        assert tools == []


class TestAgetToolSchema:
    """Tests for aget_tool_schema async method."""

    @pytest.mark.asyncio
    async def test_aget_tool_schema_uses_async_method(self) -> None:
        """Test that async method is used when available."""
        mock_router = AsyncMock()
        mock_router.aget_tool_schema = AsyncMock(return_value={"description": "Async schema"})
        router = UnifiedToolRouter(routers={"async": mock_router})
        schema = await router.aget_tool_schema("async::tool")
        assert schema["description"] == "Async schema"
        assert schema["_router"] == "async"

    @pytest.mark.asyncio
    async def test_aget_tool_schema_falls_back_to_sync(self) -> None:
        """Test fallback to sync method when async not available."""
        mock_router = MockToolRouter()
        mock_router.set_tool_schema("tool", {"description": "Sync schema"})
        router = UnifiedToolRouter(routers={"sync": mock_router})
        schema = await router.aget_tool_schema("sync::tool")
        assert schema["description"] == "Sync schema"

    @pytest.mark.asyncio
    async def test_aget_tool_schema_unknown_router(self) -> None:
        """Test getting schema from unknown router returns empty dict."""
        router = UnifiedToolRouter()
        schema = await router.aget_tool_schema("unknown::tool")
        assert schema == {}

    @pytest.mark.asyncio
    async def test_aget_tool_schema_no_default_router(self) -> None:
        """Test getting schema without default router returns empty dict."""
        router = UnifiedToolRouter()
        schema = await router.aget_tool_schema("tool")
        assert schema == {}

    @pytest.mark.asyncio
    async def test_aget_tool_schema_handles_exception(self) -> None:
        """Test that exceptions are handled gracefully."""
        mock_router = MagicMock()
        mock_router.aget_tool_schema = AsyncMock(side_effect=Exception("Error"))
        router = UnifiedToolRouter(routers={"broken": mock_router})
        schema = await router.aget_tool_schema("broken::tool")
        assert schema == {}


class TestGetToolDefinitions:
    """Tests for get_tool_definitions method."""

    def test_get_tool_definitions(self) -> None:
        """Test getting tool definitions."""
        mock_router = MockToolRouter(tools={"my_tool": lambda: None})
        mock_router.set_tool_schema(
            "my_tool",
            {
                "description": "My tool description",
                "parameters": [
                    {
                        "name": "param1",
                        "description": "First param",
                        "type": "str",
                        "required": True,
                    },
                ],
            },
        )
        router = UnifiedToolRouter(routers={"ns": mock_router})
        definitions = router.get_tool_definitions()
        assert len(definitions) == 1
        assert definitions[0].name == "ns::my_tool"
        assert definitions[0].simplified_description == "My tool description"
        assert len(definitions[0].parameters) == 1
        assert definitions[0].parameters[0].name == "param1"

    def test_get_tool_definitions_empty(self) -> None:
        """Test getting definitions from empty router."""
        router = UnifiedToolRouter()
        definitions = router.get_tool_definitions()
        assert definitions == []

    def test_get_tool_definitions_handles_exception(self) -> None:
        """Test that exceptions are handled gracefully."""
        mock_router = MagicMock()
        mock_router.get_available_tools.side_effect = Exception("Error")
        router = UnifiedToolRouter(routers={"broken": mock_router})
        definitions = router.get_tool_definitions()
        assert definitions == []

    def test_get_tool_definitions_skips_empty_schema(self) -> None:
        """Test that tools without schema are skipped."""
        mock_router = MockToolRouter(tools={"no_schema": lambda: None})
        # Don't set schema for this tool
        router = UnifiedToolRouter(routers={"ns": mock_router})
        definitions = router.get_tool_definitions()
        assert definitions == []

    def test_get_tool_definitions_with_defaults(self) -> None:
        """Test tool definitions use defaults for missing fields."""
        mock_router = MockToolRouter(tools={"tool": lambda: None})
        mock_router.set_tool_schema("tool", {"parameters": []})
        router = UnifiedToolRouter(routers={"ns": mock_router})
        definitions = router.get_tool_definitions()
        assert len(definitions) == 1
        # Should use default description
        assert "Tool tool" in definitions[0].simplified_description
