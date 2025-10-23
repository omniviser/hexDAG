"""Integration tests for mock adapters loading.

Tests demonstrate:
- Loading mock adapters from configuration
- Using mock adapters in workflows
- Registry integration with mock adapters
"""

import pytest

from hexdag.builtin.adapters.memory import InMemoryMemory
from hexdag.builtin.adapters.mock.mock_database import MockDatabaseAdapter
from hexdag.builtin.adapters.mock.mock_llm import MockLLM
from hexdag.builtin.adapters.mock.mock_tool_router import MockToolRouter
from hexdag.builtin.nodes import FunctionNode
from hexdag.core.bootstrap import bootstrap_registry
from hexdag.core.context import ExecutionContext
from hexdag.core.ports.llm import Message
from hexdag.core.registry import registry


@pytest.fixture(autouse=True)
def cleanup_registry():
    """Clean up registry before and after each test."""
    if registry.ready:
        registry._reset_for_testing()
    yield
    if registry.ready:
        registry._reset_for_testing()


class TestMockAdaptersLoading:
    """Test suite for mock adapters loading and usage."""

    def test_load_mock_adapters_from_config(self):
        """Test loading mock adapters from configuration."""
        # Bootstrap with mock config
        bootstrap_registry("hexdag/builtin/adapters/mock/hexdag.toml")

        # Check that components were loaded
        components = registry.list_components()
        mock_components = [c for c in components if "mock" in c.name or "memory" in c.name]

        assert len(mock_components) > 0

    def test_get_mock_llm_from_registry(self):
        """Test retrieving mock LLM from registry."""
        bootstrap_registry("hexdag/builtin/adapters/mock/hexdag.toml")

        llm = registry.get("mock_llm", namespace="core")
        assert llm is not None
        assert isinstance(llm, MockLLM)

    @pytest.mark.asyncio
    async def test_mock_llm_functionality(self):
        """Test mock LLM basic functionality."""
        llm = MockLLM()

        messages = [Message(role="user", content="Hello, how are you?")]
        response = await llm.aresponse(messages)

        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_mock_database_functionality(self):
        """Test mock database basic functionality."""
        db = MockDatabaseAdapter()

        # Get table schemas
        schemas = await db.aget_table_schemas()
        assert isinstance(schemas, dict)
        assert len(schemas) > 0

        # Execute query
        results = await db.aexecute_query("SELECT * FROM customers LIMIT 2")
        assert results is not None

    @pytest.mark.asyncio
    async def test_in_memory_memory_functionality(self):
        """Test in-memory memory adapter functionality."""
        memory = InMemoryMemory()

        # Set value
        await memory.aset("test_key", {"data": "value"})

        # Get value
        result = await memory.aget("test_key")
        assert result == {"data": "value"}

    @pytest.mark.asyncio
    async def test_mock_tool_router_functionality(self):
        """Test mock tool router functionality."""
        router = MockToolRouter()

        # Get available tools
        tools = router.get_available_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

        # Call tool
        result = await router.acall_tool("calculate", {"expression": "42 * 2"})
        assert result is not None

    @pytest.mark.asyncio
    async def test_workflow_with_mock_adapters(self):
        """Test complete workflow using mock adapters."""

        async def process_customer_query(input_data, **kwargs):
            """Process a customer query using mock adapters."""
            from hexdag.core.context import get_port

            llm = get_port("llm")
            db = get_port("database")
            memory = get_port("memory")
            tools = get_port("tool_router")

            # Check memory for cached response
            cached = await memory.aget(f"query_{input_data['query_id']}")
            if cached:
                return {"source": "cache", "response": cached}

            # Query database for customer info
            customer_data = await db.aexecute_query(
                "SELECT * FROM customers WHERE id = ?", {"id": input_data["customer_id"]}
            )

            # Use tool to get additional info
            weather = await tools.acall_tool(
                "get_weather", {"location": input_data.get("location", "Unknown")}
            )

            # Generate response with LLM
            messages = [
                Message(
                    role="system",
                    content="You are a helpful customer service assistant.",
                ),
                Message(
                    role="user",
                    content=f"Customer {customer_data}: {input_data['query']}. Weather: {weather}",
                ),
            ]
            response = await llm.aresponse(messages)

            # Cache the response
            await memory.aset(f"query_{input_data['query_id']}", response)

            return {
                "source": "generated",
                "response": response,
                "customer_data": customer_data,
                "weather": weather,
            }

        # Create the node
        node_factory = FunctionNode()
        node = node_factory(
            name="process_query",
            fn=process_customer_query,
            deps=[],
        )

        # Create adapter instances
        ports = {
            "llm": MockLLM(),
            "database": MockDatabaseAdapter(),
            "memory": InMemoryMemory(),
            "tool_router": MockToolRouter(),
        }

        # Execute the node
        input_data = {
            "query_id": "Q123",
            "customer_id": 1,
            "query": "What's the status of my order?",
            "location": "New York",
        }

        async with ExecutionContext(ports=ports):
            result = await node.fn(input_data)

        assert result["source"] == "generated"
        assert "response" in result
        assert "weather" in result

    @pytest.mark.asyncio
    async def test_mock_adapter_caching(self):
        """Test that mock adapters support caching."""

        async def cached_operation(input_data, **kwargs):
            from hexdag.core.context import get_port

            memory = get_port("memory")

            # Try to get cached value
            cached = await memory.aget("test_key")
            if cached:
                return {"source": "cache", "value": cached}

            # Compute and cache
            value = {"computed": input_data * 2}
            await memory.aset("test_key", value)

            return {"source": "computed", "value": value}

        node_factory = FunctionNode()
        node = node_factory(name="cached_op", fn=cached_operation, deps=[])

        ports = {"memory": InMemoryMemory()}

        # First call - should compute
        async with ExecutionContext(ports=ports):
            result1 = await node.fn(42)
        assert result1["source"] == "computed"

        # Second call - should use cache
        async with ExecutionContext(ports=ports):
            result2 = await node.fn(42)
        assert result2["source"] == "cache"
