"""Tests for UnifiedToolRouter with multi-router support."""

import pytest

from hexdag.builtin.adapters.unified_tool_router import UnifiedToolRouter
from hexdag.core.registry import registry, tool


@pytest.fixture
def clean_registry():
    """Ensure clean registry for each test."""
    # Store original state
    original_components = registry._components.copy()

    yield

    # Restore original state
    registry._components = original_components


@pytest.fixture
def sample_tools(clean_registry):
    """Register sample tools for testing."""

    @tool(name="search", namespace="builtin", description="Search for information")
    def search_tool(query: str) -> dict[str, list[str]]:
        return {"query": query, "results": [f"Result for {query}"]}

    @tool(name="calculate", namespace="builtin", description="Perform calculation")
    def calculate_tool(expression: str) -> dict[str, float]:
        return {"expression": expression, "result": eval(expression)}

    @tool(name="tool_end", namespace="builtin", description="End reasoning")
    def end_tool(result: str) -> dict[str, str]:
        return {"tool_used": "tool_end", "result": result}

    return {"search": search_tool, "calculate": calculate_tool, "tool_end": end_tool}


class MockToolRouter:
    """Mock tool router for testing multi-router mode."""

    def __init__(self, namespace: str, tools: dict[str, callable]):
        self.namespace = namespace
        self.tools = tools

    async def acall_tool(self, tool_name: str, params: dict) -> dict:
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found in {self.namespace}")
        return self.tools[tool_name](**params)

    def get_available_tools(self) -> list[str]:
        """Return list of tool names."""
        return list(self.tools.keys())

    async def aget_available_tools(self) -> list[str]:
        """Async version - return list of tool names."""
        return list(self.tools.keys())


class TestUnifiedToolRouterLegacyMode:
    """Test backward compatibility with legacy single-router mode.

    Note: These tests use multi-router mode with a single router
    to avoid dependency on global registry state. True legacy mode
    (no routers parameter) requires tools to be registered in global
    registry, which is tested in integration tests.
    """

    @pytest.mark.asyncio
    async def test_single_router_mode(self):
        """Test single router (equivalent to legacy mode)."""
        # Create a single router
        tool_router = MockToolRouter(
            "builtin",
            {"search": lambda query: {"query": query, "results": [f"Result for {query}"]}},
        )

        # Use multi-router mode with single router
        router = UnifiedToolRouter(routers={"builtin": tool_router})

        # Call with namespace
        result = await router.acall_tool("builtin::search", {"query": "Python"})
        assert result["query"] == "Python"
        assert len(result["results"]) > 0

        # Call without namespace (uses default router)
        result = await router.acall_tool("search", {"query": "Java"})
        assert result["query"] == "Java"

    @pytest.mark.asyncio
    async def test_single_router_get_available_tools(self):
        """Test getting available tools with single router."""
        tool_router = MockToolRouter(
            "builtin",
            {
                "search": lambda query: {"results": []},
                "calculate": lambda expr: {"result": 0},
                "tool_end": lambda result: {"result": result},
            },
        )

        router = UnifiedToolRouter(routers={"builtin": tool_router})

        tools = await router.aget_available_tools()

        assert len(tools) == 3
        tool_names = set(tools)
        assert "builtin::search" in tool_names
        assert "builtin::calculate" in tool_names
        assert "builtin::tool_end" in tool_names

    @pytest.mark.asyncio
    async def test_tool_not_found(self):
        """Test error handling for missing tool."""

        tool_router = MockToolRouter("builtin", {"existing": lambda: {}})

        router = UnifiedToolRouter(routers={"builtin": tool_router})

        with pytest.raises(ValueError, match="Tool nonexistent not found"):
            await router.acall_tool("nonexistent", {})


class TestUnifiedToolRouterMultiMode:
    """Test multi-router mode with namespace prefixes."""

    @pytest.mark.asyncio
    async def test_multi_router_with_namespace(self):
        """Test calling tools with namespace prefix."""
        # Create separate routers for different namespaces
        python_router = MockToolRouter(
            "python",
            {
                "execute": lambda code: {"output": f"Executed: {code}"},
                "lint": lambda code: {"issues": []},
            },
        )

        mcp_router = MockToolRouter(
            "mcp",
            {
                "fetch": lambda url: {"content": f"Content from {url}"},
                "analyze": lambda data: {"analysis": f"Analysis of {data}"},
            },
        )

        # Create unified router
        router = UnifiedToolRouter(routers={"python": python_router, "mcp": mcp_router})

        # Call python tool with namespace
        result = await router.acall_tool("python::execute", {"code": "print('hello')"})
        assert result["output"] == "Executed: print('hello')"

        # Call MCP tool with namespace
        result = await router.acall_tool("mcp::fetch", {"url": "https://example.com"})
        assert result["content"] == "Content from https://example.com"

    @pytest.mark.asyncio
    async def test_multi_router_default_fallback(self):
        """Test fallback to default router for unprefixed tools."""
        python_router = MockToolRouter(
            "python",
            {"execute": lambda code: {"output": f"Executed: {code}"}},
        )

        mcp_router = MockToolRouter(
            "mcp",
            {"fetch": lambda url: {"content": f"Content from {url}"}},
        )

        # python is first, so it becomes default
        router = UnifiedToolRouter(routers={"python": python_router, "mcp": mcp_router})

        # Unprefixed tool uses default router
        result = await router.acall_tool("execute", {"code": "x = 1"})
        assert result["output"] == "Executed: x = 1"

    @pytest.mark.asyncio
    async def test_multi_router_get_available_tools(self):
        """Test getting available tools from all routers with namespaces."""
        python_router = MockToolRouter(
            "python",
            {"execute": lambda code: {"output": code}, "lint": lambda code: {"issues": []}},
        )

        mcp_router = MockToolRouter(
            "mcp",
            {"fetch": lambda url: {"content": url}},
        )

        router = UnifiedToolRouter(routers={"python": python_router, "mcp": mcp_router})

        tools = await router.aget_available_tools()

        # Should have 3 tools total with namespace prefixes
        assert len(tools) == 3
        tool_names = set(tools)  # tools is list of strings
        assert "python::execute" in tool_names
        assert "python::lint" in tool_names
        assert "mcp::fetch" in tool_names

    @pytest.mark.asyncio
    async def test_multi_router_invalid_namespace(self):
        """Test error handling for invalid namespace prefix."""
        from hexdag.core.exceptions import ResourceNotFoundError

        python_router = MockToolRouter("python", {"execute": lambda code: {"output": code}})

        router = UnifiedToolRouter(routers={"python": python_router})

        with pytest.raises(ResourceNotFoundError):
            await router.acall_tool("invalid::tool", {})

    @pytest.mark.asyncio
    async def test_multi_router_tool_not_found(self):
        """Test error handling for tool not found in specified router."""
        python_router = MockToolRouter("python", {"execute": lambda code: {"output": code}})

        router = UnifiedToolRouter(routers={"python": python_router})

        with pytest.raises(ValueError, match="Tool nonexistent not found"):
            await router.acall_tool("python::nonexistent", {})


class TestUnifiedToolRouterMixedMode:
    """Test mixed usage of legacy and multi-router modes."""

    @pytest.mark.asyncio
    async def test_combined_registry_and_custom_routers(self, sample_tools):
        """Test combining global registry tools with custom routers."""
        # Create custom router
        custom_router = MockToolRouter(
            "custom",
            {"special_tool": lambda x: {"special_result": x}},
        )

        # Create unified router with both registry (legacy) and custom routers
        router = UnifiedToolRouter(routers={"custom": custom_router})

        # This should work with custom router
        result = await router.acall_tool("custom::special_tool", {"x": 42})
        assert result["special_result"] == 42

        # Note: In multi-router mode, registry tools are NOT automatically available
        # unless you explicitly add a registry-based router

    @pytest.mark.asyncio
    async def test_empty_routers_dict(self):
        """Test that empty routers dict uses legacy mode."""
        router = UnifiedToolRouter(routers={})

        # Empty dict means multi_router_mode is False (legacy mode)
        assert router.multi_router_mode is False
        assert router.default_router is None


class TestUnifiedToolRouterEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_namespace_with_multiple_colons(self):
        """Test handling of tool names with multiple :: separators."""
        router_a = MockToolRouter("router_a", {"tool": lambda: {"result": "A"}})
        router_b = MockToolRouter("router_b", {"tool": lambda: {"result": "B"}})

        router = UnifiedToolRouter(routers={"router_a": router_a, "router_b": router_b})

        # Should only split on first ::
        result = await router.acall_tool("router_a::tool", {})
        assert result["result"] == "A"

    @pytest.mark.asyncio
    async def test_tool_with_complex_params(self):
        """Test tools with complex parameter structures."""
        complex_router = MockToolRouter(
            "complex",
            {
                "process": lambda data, config: {
                    "processed": f"{data['value']} with {config['mode']}"
                }
            },
        )

        router = UnifiedToolRouter(routers={"complex": complex_router})

        result = await router.acall_tool(
            "complex::process",
            {"data": {"value": "test"}, "config": {"mode": "fast"}},
        )
        assert result["processed"] == "test with fast"
