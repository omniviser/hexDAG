"""Tests for MockToolRouter implementation."""

import pytest

from hexai.adapters.configs import MockToolRouterConfig
from hexai.adapters.mock import MockToolRouter


class TestMockToolRouter:
    """Test cases for MockToolRouter."""

    @pytest.fixture
    def router(self):
        """Create a MockToolRouter instance for testing."""
        return MockToolRouter()

    @pytest.mark.asyncio
    async def test_default_tools_available(self, router):
        """Test that default mock tools are available."""
        tools = router.get_available_tools()
        assert "search" in tools
        assert "calculate" in tools
        assert "get_weather" in tools

    @pytest.mark.asyncio
    async def test_search_tool(self, router):
        """Test the search tool functionality."""
        result = await router.acall_tool("search", {"query": "test query"})
        assert "results" in result
        assert len(result["results"]) == 2
        assert "test query" in result["results"][0]["title"]

    @pytest.mark.asyncio
    async def test_calculate_tool(self, router):
        """Test the calculate tool functionality."""
        result = await router.acall_tool("calculate", {"expression": "2 + 2"})
        assert result == {"result": 4}

        # Test with invalid expression
        result = await router.acall_tool("calculate", {"expression": "invalid"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_weather_tool(self, router):
        """Test the weather tool functionality."""
        result = await router.acall_tool("get_weather", {"location": "New York"})
        assert result["location"] == "New York"
        assert "temperature" in result
        assert "conditions" in result
        assert "humidity" in result

    @pytest.mark.asyncio
    async def test_custom_tools_from_config(self):
        """Test adding custom tools via configuration."""
        config = MockToolRouterConfig(
            available_tools=["custom_tool1", "custom_tool2"]
        )
        router = MockToolRouter(config)

        tools = router.get_available_tools()
        assert "custom_tool1" in tools
        assert "custom_tool2" in tools

        # Test calling custom tool
        result = await router.acall_tool("custom_tool1", {"param": "value"})
        assert result["tool"] == "custom_tool1"
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_unknown_tool_raises_error(self):
        """Test that unknown tools raise an error when configured to do so."""
        config = MockToolRouterConfig(raise_on_unknown_tool=True)
        router = MockToolRouter(config)

        with pytest.raises(ValueError, match="Unknown tool: nonexistent"):
            await router.acall_tool("nonexistent", {})

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        """Test that unknown tools return an error when configured not to raise."""
        config = MockToolRouterConfig(raise_on_unknown_tool=False)
        router = MockToolRouter(config)

        result = await router.acall_tool("nonexistent", {})
        assert "error" in result
        assert "Unknown tool: nonexistent" in result["error"]

    @pytest.mark.asyncio
    async def test_delay_simulation(self):
        """Test that delay simulation works."""
        import time

        config = MockToolRouterConfig(delay_seconds=0.1)
        router = MockToolRouter(config)

        start_time = time.time()
        await router.acall_tool("search", {"query": "test"})
        elapsed_time = time.time() - start_time

        assert elapsed_time >= 0.1

    def test_get_tool_schema(self, router):
        """Test getting schema for a specific tool."""
        schema = router.get_tool_schema("search")
        assert schema["name"] == "search"
        assert "description" in schema
        assert "parameters" in schema
        assert "query" in schema["parameters"]

        # Test unknown tool
        with pytest.raises(KeyError, match="Tool not found: nonexistent"):
            router.get_tool_schema("nonexistent")

    def test_get_all_tool_schemas(self, router):
        """Test getting schemas for all tools."""
        schemas = router.get_all_tool_schemas()
        assert "search" in schemas
        assert "calculate" in schemas
        assert "get_weather" in schemas

        for tool_name, schema in schemas.items():
            assert schema["name"] == tool_name
            assert "description" in schema
            assert "parameters" in schema

    @pytest.mark.asyncio
    async def test_call_history_tracking(self, router):
        """Test that call history is tracked properly."""
        await router.acall_tool("search", {"query": "test1"})
        await router.acall_tool("calculate", {"expression": "1+1"})

        history = router.get_call_history()
        assert len(history) == 2
        assert history[0]["tool"] == "search"
        assert history[0]["params"]["query"] == "test1"
        assert history[1]["tool"] == "calculate"
        assert history[1]["params"]["expression"] == "1+1"

        # Test reset
        router.reset()
        assert len(router.get_call_history()) == 0

    def test_add_tool_dynamically(self, router):
        """Test adding new tools dynamically."""
        router.add_tool(
            "dynamic_tool",
            "A dynamically added tool",
            {"input": {"type": "string", "description": "Input parameter"}}
        )

        tools = router.get_available_tools()
        assert "dynamic_tool" in tools

        schema = router.get_tool_schema("dynamic_tool")
        assert schema["name"] == "dynamic_tool"
        assert schema["description"] == "A dynamically added tool"
