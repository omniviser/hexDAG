"""Tests for ToolCallNode - parallel tool execution with port injection."""

import pytest

from hexdag.builtin.nodes.tool_call_node import ToolCallNode
from hexdag.core.domain.dag import DirectedGraph
from hexdag.core.orchestration.hooks import HookConfig
from hexdag.core.orchestration.orchestrator import Orchestrator
from hexdag.core.registry import registry
from hexdag.core.registry.models import ComponentMetadata, ComponentType, FunctionComponent


def register_tool(
    name: str,
    fn,
    namespace: str = "test",
    port_requirements: list[str] | None = None,
):
    """Helper to manually register a tool in the registry."""
    # Registry stores components by name only (not qualified name like "namespace:name")
    registry._components[name] = ComponentMetadata(
        name=name,
        namespace=namespace,
        component_type=ComponentType.TOOL,
        component=FunctionComponent(value=fn),  # FunctionComponent uses "value" not "fn"
        description=fn.__doc__ or "",
        port_requirements=port_requirements or [],
    )


@pytest.fixture(autouse=True)
def cleanup_test_tools():
    """Clean up test tools from registry after each test."""
    # Store original components before test
    original_keys = set(registry._components.keys())
    yield
    # Remove any new components added during test
    current_keys = set(registry._components.keys())
    new_keys = current_keys - original_keys
    for key in new_keys:
        del registry._components[key]


@pytest.mark.asyncio
async def test_tool_call_node_basic():
    """Test ToolCallNode executes a simple tool."""

    # Define and register a simple tool
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    register_tool("add_numbers", add_numbers)

    # Create ToolCallNode
    tool_node_factory = ToolCallNode()
    tool_spec = tool_node_factory(
        name="add_tool",
        tool_name="test:add_numbers",
        arguments={"a": 5, "b": 3},
    )

    # Execute in graph
    graph = DirectedGraph()
    graph += tool_spec

    orchestrator = Orchestrator()
    results = await orchestrator.run(graph, {})

    # Verify result (results["add_tool"] is a ToolCallOutput Pydantic model)
    assert results["add_tool"].result == 8  # .result is the tool output
    assert results["add_tool"].tool_name == "test:add_numbers"
    assert results["add_tool"].error is None


@pytest.mark.asyncio
async def test_tool_call_node_async_tool():
    """Test ToolCallNode executes async tools."""

    # Register async tool
    async def async_multiply(x: int, y: int) -> int:
        """Multiply two numbers asynchronously."""
        return x * y

    register_tool("async_multiply", async_multiply)

    # Create ToolCallNode
    tool_node_factory = ToolCallNode()
    tool_spec = tool_node_factory(
        name="multiply_tool",
        tool_name="test:async_multiply",
        arguments={"x": 4, "y": 7},
    )

    # Execute
    graph = DirectedGraph()
    graph += tool_spec

    orchestrator = Orchestrator()
    results = await orchestrator.run(graph, {})

    # Verify
    assert results["multiply_tool"].result == 28


@pytest.mark.asyncio
async def test_tool_call_node_with_port_injection():
    """Test ToolCallNode injects required ports automatically."""
    from hexdag.builtin.adapters.mock.mock_database import MockDatabaseAdapter

    # Register tool that requires database port
    async def query_db(sql: str, database_port=None) -> list[dict]:
        """Execute SQL query using database port."""
        if database_port is None:
            raise ValueError("database_port is required")
        return await database_port.aexecute_query(sql, {})

    register_tool("query_db", query_db, port_requirements=["database"])

    # Create ToolCallNode
    tool_node_factory = ToolCallNode()
    tool_spec = tool_node_factory(
        name="db_query_tool",
        tool_name="test:query_db",
        arguments={"sql": "SELECT * FROM users"},
    )

    # Execute with database port
    graph = DirectedGraph()
    graph += tool_spec

    # Configure mock database
    mock_db = MockDatabaseAdapter(enable_sample_data=False)

    # Disable health checks for mock adapters
    hook_config = HookConfig(enable_health_checks=False)
    orchestrator = Orchestrator(ports={"database": mock_db}, pre_hook_config=hook_config)
    results = await orchestrator.run(graph, {})

    # Verify database port was injected and used
    assert results["db_query_tool"].error is None
    # MockDatabaseAdapter returns sample data by default
    assert isinstance(results["db_query_tool"].result, list)


@pytest.mark.asyncio
async def test_tool_call_node_multiple_ports():
    """Test ToolCallNode injects multiple ports."""
    from hexdag.builtin.adapters.mock.mock_database import MockDatabaseAdapter
    from hexdag.builtin.adapters.mock.mock_llm import MockLLM
    from hexdag.core.ports.llm import Message

    # Register tool needing both database and LLM
    async def semantic_search(
        query: str,
        database_port=None,
        llm_port=None,
    ) -> dict:
        """Perform semantic search using LLM embeddings and database."""
        if not database_port or not llm_port:
            raise ValueError("Both database_port and llm_port required")

        # Use LLM for embedding (mock)
        embedding_response = await llm_port.aresponse([
            Message(role="user", content=f"embed: {query}")
        ])

        # Query database with embedding
        results = await database_port.aexecute_query(
            "SELECT * FROM docs WHERE embedding = ?",
            {"embedding": embedding_response},
        )

        return {"query": query, "results": results, "embedding": embedding_response}

    register_tool("semantic_search", semantic_search, port_requirements=["database", "llm"])

    # Create ToolCallNode
    tool_node_factory = ToolCallNode()
    tool_spec = tool_node_factory(
        name="search_tool",
        tool_name="test:semantic_search",
        arguments={"query": "machine learning"},
    )

    # Execute with both ports
    graph = DirectedGraph()
    graph += tool_spec

    mock_db = MockDatabaseAdapter(enable_sample_data=False)
    mock_llm = MockLLM(responses=['{"embedding": [0.1, 0.2, 0.3]}'])

    # Disable health checks for mock adapters
    hook_config = HookConfig(enable_health_checks=False)
    orchestrator = Orchestrator(
        ports={
            "database": mock_db,
            "llm": mock_llm,
        },
        pre_hook_config=hook_config,
    )

    results = await orchestrator.run(graph, {})

    # Verify both ports were injected
    result = results["search_tool"].result  # This is the dict returned by tool
    assert result["query"] == "machine learning"
    assert "embedding" in result
    assert isinstance(result["results"], list)


@pytest.mark.asyncio
async def test_tool_call_node_missing_port():
    """Test ToolCallNode handles missing port gracefully."""

    # Register tool requiring port
    async def needs_db(value: str, database_port=None) -> str:
        """Tool that checks if database port exists."""
        if database_port is None:
            return "no_port"
        return "has_port"

    register_tool("needs_db", needs_db, port_requirements=["database"])

    # Create ToolCallNode
    tool_node_factory = ToolCallNode()
    tool_spec = tool_node_factory(
        name="db_check_tool",
        tool_name="test:needs_db",
        arguments={"value": "test"},
    )

    # Execute WITHOUT providing database port
    graph = DirectedGraph()
    graph += tool_spec

    orchestrator = Orchestrator()  # No ports!
    results = await orchestrator.run(graph, {})

    # Tool should receive None and handle gracefully
    assert results["db_check_tool"].result == "no_port"


@pytest.mark.asyncio
async def test_parallel_tool_execution():
    """Test multiple ToolCallNodes execute in parallel."""
    import asyncio
    import time

    # Register tools with delay
    async def slow_tool_1(value: int) -> int:
        """Slow tool 1."""
        await asyncio.sleep(0.1)
        return value * 2

    async def slow_tool_2(value: int) -> int:
        """Slow tool 2."""
        await asyncio.sleep(0.1)
        return value * 3

    register_tool("slow_tool_1", slow_tool_1)
    register_tool("slow_tool_2", slow_tool_2)

    # Create two ToolCallNodes
    tool_factory = ToolCallNode()

    tool1 = tool_factory(
        name="tool_1",
        tool_name="test:slow_tool_1",
        arguments={"value": 5},
    )

    tool2 = tool_factory(
        name="tool_2",
        tool_name="test:slow_tool_2",
        arguments={"value": 5},
    )

    # Execute in parallel
    graph = DirectedGraph()
    graph += tool1
    graph += tool2

    start_time = time.time()
    orchestrator = Orchestrator()
    results = await orchestrator.run(graph, {})
    duration = time.time() - start_time

    # Verify parallel execution (should be ~0.1s, not 0.2s)
    assert duration < 0.15, f"Tools should run in parallel, took {duration}s"
    assert results["tool_1"].result == 10
    assert results["tool_2"].result == 15


@pytest.mark.asyncio
async def test_tool_call_node_error_handling():
    """Test ToolCallNode handles tool errors gracefully."""

    # Register tool that raises error
    def failing_tool(value: str) -> str:
        """Tool that always fails."""
        raise ValueError(f"Intentional error: {value}")

    register_tool("failing_tool", failing_tool)

    # Create ToolCallNode
    tool_node_factory = ToolCallNode()
    tool_spec = tool_node_factory(
        name="fail_tool",
        tool_name="test:failing_tool",
        arguments={"value": "test"},
        tool_call_id="call_123",
    )

    # Execute
    graph = DirectedGraph()
    graph += tool_spec

    orchestrator = Orchestrator()
    results = await orchestrator.run(graph, {})

    # Verify error captured, not raised
    assert results["fail_tool"].result is None
    assert results["fail_tool"].error is not None
    assert "Intentional error" in results["fail_tool"].error
    assert results["fail_tool"].tool_call_id == "call_123"


@pytest.mark.asyncio
async def test_tool_call_id_preservation():
    """Test tool_call_id is preserved through execution."""

    def simple_tool(x: int) -> int:
        return x + 1

    register_tool("simple_tool", simple_tool)

    # Create ToolCallNode with call ID
    tool_node_factory = ToolCallNode()
    tool_spec = tool_node_factory(
        name="id_test_tool",
        tool_name="test:simple_tool",
        arguments={"x": 10},
        tool_call_id="call_xyz_789",
    )

    # Execute
    graph = DirectedGraph()
    graph += tool_spec

    orchestrator = Orchestrator()
    results = await orchestrator.run(graph, {})

    # Verify tool_call_id preserved
    assert results["id_test_tool"].tool_call_id == "call_xyz_789"
