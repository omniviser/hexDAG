"""Tests for MockToolAdapter."""

import pytest

from hexdag.builtin.adapters.mock import MockToolAdapter


class TestMockToolAdapter:
    """Test cases for MockToolAdapter."""

    @pytest.fixture
    def basic_mock_tools(self):
        """Basic mock tools for testing."""
        return {
            "search_customers": [
                {"id": 1, "name": "Alice", "email": "alice@example.com"},
                {"id": 2, "name": "Bob", "email": "bob@example.com"},
            ],
            "get_product": {
                "id": 42,
                "name": "Widget",
                "price": 9.99,
                "in_stock": True,
            },
            "calculate_tax": 0.08,  # Simple value response
        }

    def test_initialization_with_mock_responses(self, basic_mock_tools):
        """Test adapter initialization with predefined responses."""
        adapter = MockToolAdapter(mock_responses=basic_mock_tools)

        assert adapter.get_available_tools() == ["search_customers", "get_product", "calculate_tax"]
        assert len(adapter.call_history) == 0

    def test_initialization_without_responses(self):
        """Test adapter initialization without responses."""
        adapter = MockToolAdapter()

        assert adapter.get_available_tools() == []
        assert adapter.mock_responses == {}

    @pytest.mark.asyncio
    async def test_call_tool_with_predefined_response(self, basic_mock_tools):
        """Test calling a tool returns its predefined response."""
        adapter = MockToolAdapter(mock_responses=basic_mock_tools)

        # Test list response
        result = await adapter.acall_tool("search_customers", {"query": "test"})
        assert result == [
            {"id": 1, "name": "Alice", "email": "alice@example.com"},
            {"id": 2, "name": "Bob", "email": "bob@example.com"},
        ]

        # Test dict response
        result = await adapter.acall_tool("get_product", {"id": 42})
        assert result == {
            "id": 42,
            "name": "Widget",
            "price": 9.99,
            "in_stock": True,
        }

        # Test simple value response
        result = await adapter.acall_tool("calculate_tax", {"amount": 100})
        assert result == 0.08

    @pytest.mark.asyncio
    async def test_call_history_tracking(self, basic_mock_tools):
        """Test that call history is properly tracked."""
        adapter = MockToolAdapter(mock_responses=basic_mock_tools)

        await adapter.acall_tool("search_customers", {"query": "Alice"})
        await adapter.acall_tool("get_product", {"id": 42})

        assert len(adapter.call_history) == 2
        assert adapter.call_history[0] == {
            "tool": "search_customers",
            "params": {"query": "Alice"},
        }
        assert adapter.call_history[1] == {
            "tool": "get_product",
            "params": {"id": 42},
        }

    @pytest.mark.asyncio
    async def test_callable_response(self):
        """Test that callable responses are executed with params."""

        def dynamic_response(params):
            """Generate response based on params."""
            count = params.get("count", 1)
            return [{"id": i, "value": i * 10} for i in range(count)]

        adapter = MockToolAdapter(mock_responses={"generate_data": dynamic_response})

        result = await adapter.acall_tool("generate_data", {"count": 3})
        assert result == [
            {"id": 0, "value": 0},
            {"id": 1, "value": 10},
            {"id": 2, "value": 20},
        ]

    @pytest.mark.asyncio
    async def test_unknown_tool_with_default_response(self):
        """Test unknown tool returns default response."""
        adapter = MockToolAdapter(
            mock_responses={"known_tool": "known"},
            default_response={"error": "Tool not implemented"},
        )

        result = await adapter.acall_tool("unknown_tool", {})
        assert result == {"error": "Tool not implemented"}

    @pytest.mark.asyncio
    async def test_unknown_tool_without_default_response(self):
        """Test unknown tool returns standard mock response."""
        adapter = MockToolAdapter(mock_responses={"known_tool": "known"})

        result = await adapter.acall_tool("unknown_tool", {})
        assert result == {
            "status": "success",
            "tool": "unknown_tool",
            "message": "Mock response for unknown_tool",
        }

    @pytest.mark.asyncio
    async def test_unknown_tool_raises_error(self):
        """Test unknown tool raises error when configured."""
        adapter = MockToolAdapter(
            mock_responses={"known_tool": "known"},
            raise_on_unknown=True,
        )

        with pytest.raises(ValueError, match="Unknown tool: 'unknown_tool'"):
            await adapter.acall_tool("unknown_tool", {})

    def test_get_tool_schema(self, basic_mock_tools):
        """Test getting schema for a tool."""
        adapter = MockToolAdapter(mock_responses=basic_mock_tools)

        schema = adapter.get_tool_schema("search_customers")
        assert schema == {
            "name": "search_customers",
            "description": "Mock tool: search_customers",
            "parameters": {},
        }

        # Unknown tool returns empty schema
        schema = adapter.get_tool_schema("unknown")
        assert schema == {}

    def test_get_all_tool_schemas(self, basic_mock_tools):
        """Test getting schemas for all tools."""
        adapter = MockToolAdapter(mock_responses=basic_mock_tools)

        schemas = adapter.get_all_tool_schemas()
        assert len(schemas) == 3
        assert "search_customers" in schemas
        assert "get_product" in schemas
        assert "calculate_tax" in schemas

    def test_set_response(self):
        """Test dynamically setting tool responses."""
        adapter = MockToolAdapter()

        # Initially no tools
        assert adapter.get_available_tools() == []

        # Add a tool
        adapter.set_response("new_tool", {"result": "success"})
        assert "new_tool" in adapter.get_available_tools()

        # Update existing tool
        adapter.set_response("new_tool", {"result": "updated"})
        assert len(adapter.get_available_tools()) == 1

    @pytest.mark.asyncio
    async def test_remove_tool(self):
        """Test removing a tool."""
        adapter = MockToolAdapter(mock_responses={"tool1": "response1", "tool2": "response2"})

        assert len(adapter.get_available_tools()) == 2

        adapter.remove_tool("tool1")
        assert adapter.get_available_tools() == ["tool2"]

        # Removing non-existent tool doesn't raise error
        adapter.remove_tool("non_existent")
        assert adapter.get_available_tools() == ["tool2"]

    @pytest.mark.asyncio
    async def test_clear_history(self):
        """Test clearing call history."""
        adapter = MockToolAdapter(mock_responses={"tool": "response"})

        await adapter.acall_tool("tool", {})
        await adapter.acall_tool("tool", {})
        assert len(adapter.call_history) == 2

        adapter.clear_history()
        assert len(adapter.call_history) == 0

    @pytest.mark.asyncio
    async def test_get_call_count(self):
        """Test getting call counts."""
        adapter = MockToolAdapter(mock_responses={"tool1": "r1", "tool2": "r2"})

        await adapter.acall_tool("tool1", {})
        await adapter.acall_tool("tool2", {})
        await adapter.acall_tool("tool1", {})

        # Total calls
        assert adapter.get_call_count() == 3

        # Specific tool calls
        assert adapter.get_call_count("tool1") == 2
        assert adapter.get_call_count("tool2") == 1
        assert adapter.get_call_count("tool3") == 0

    @pytest.mark.asyncio
    async def test_get_last_call(self):
        """Test getting last call details."""
        adapter = MockToolAdapter(mock_responses={"tool1": "r1", "tool2": "r2"})

        # No calls yet
        assert adapter.get_last_call() is None
        assert adapter.get_last_call("tool1") is None

        await adapter.acall_tool("tool1", {"param": "a"})
        await adapter.acall_tool("tool2", {"param": "b"})
        await adapter.acall_tool("tool1", {"param": "c"})

        # Last call overall
        last = adapter.get_last_call()
        assert last["tool"] == "tool1"
        assert last["params"] == {"param": "c"}

        # Last call for specific tool
        last_tool1 = adapter.get_last_call("tool1")
        assert last_tool1["params"] == {"param": "c"}

        last_tool2 = adapter.get_last_call("tool2")
        assert last_tool2["params"] == {"param": "b"}

        # No calls for unknown tool
        assert adapter.get_last_call("tool3") is None

    @pytest.mark.asyncio
    async def test_complex_nested_response(self):
        """Test with complex nested data structures."""
        complex_response = {
            "users": [
                {
                    "id": 1,
                    "profile": {
                        "name": "Alice",
                        "settings": {"theme": "dark", "notifications": True},
                    },
                    "orders": [
                        {"id": 101, "total": 99.99},
                        {"id": 102, "total": 49.99},
                    ],
                }
            ],
            "metadata": {
                "total_count": 1,
                "page": 1,
            },
        }

        adapter = MockToolAdapter(mock_responses={"get_user_data": complex_response})

        result = await adapter.acall_tool("get_user_data", {"user_id": 1})
        assert result == complex_response
        assert result["users"][0]["profile"]["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_none_response(self):
        """Test that None can be a valid response."""
        adapter = MockToolAdapter(mock_responses={"check_exists": None})

        result = await adapter.acall_tool("check_exists", {"id": 123})
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_responses(self):
        """Test various empty response types."""
        adapter = MockToolAdapter(
            mock_responses={
                "empty_list": [],
                "empty_dict": {},
                "empty_string": "",
                "zero": 0,
                "false": False,
            }
        )

        assert await adapter.acall_tool("empty_list", {}) == []
        assert await adapter.acall_tool("empty_dict", {}) == {}
        assert await adapter.acall_tool("empty_string", {}) == ""
        assert await adapter.acall_tool("zero", {}) == 0
        assert await adapter.acall_tool("false", {}) is False

    def test_example_usage_from_docstring(self):
        """Test the example usage from the class docstring."""
        mock_tools = {
            "search_customers": [{"id": 1, "name": "Alice"}],
            "get_product": {"id": 42, "name": "Widget", "price": 9.99},
        }
        router = MockToolAdapter(mock_tools)

        assert router.get_available_tools() == ["search_customers", "get_product"]
