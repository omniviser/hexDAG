#!/usr/bin/env python3
"""Example demonstrating mock adapters with HexDAG configuration.

This example shows how the mock adapters are loaded as plugins via hexdag.toml
and can be used in workflows for testing and development.
"""

import asyncio

from hexai.core.bootstrap import bootstrap_registry
from hexai.core.registry import registry


async def demo_mock_adapters():
    """Demonstrate using mock adapters loaded from configuration."""
    print("=" * 60)
    print("Mock Adapters Demo - Loading with Mock Config")
    print("=" * 60)

    # Bootstrap registry with mock config to load mock adapters for testing
    bootstrap_registry("hexai/adapters/mock/hexdag.toml")

    print("\n📦 Loaded Components:")
    components = registry.list_components()
    for comp in components:
        if "mock" in comp.name or "memory" in comp.name:
            print(f"  - {comp.qualified_name}: {comp.component_type.value}")

    # Get mock adapters from registry
    print("\n🔌 Using Mock Adapters:")

    # 1. Mock LLM
    llm = registry.get("mock_llm", namespace="plugin")
    print("\n1. Mock LLM:")
    from hexai.core.ports.llm import Message

    messages = [Message(role="user", content="Hello, how are you?")]
    response = await llm.aresponse(messages)
    print(f"   User: {messages[0].content}")
    print(f"   Assistant: {response}")

    # 2. Mock Database
    db = registry.get("mock_database", namespace="plugin")
    print("\n2. Mock Database:")
    schemas = await db.aget_table_schemas()
    print(f"   Available tables: {list(schemas.keys())}")

    results = await db.aexecute_query("SELECT * FROM customers LIMIT 2")
    print(f"   Sample query results: {results}")

    # 3. In-Memory Memory
    memory = registry.get("in_memory_memory", namespace="plugin")
    print("\n3. In-Memory Memory:")
    await memory.aset("user_preference", {"theme": "dark", "language": "en"})
    pref = await memory.aget("user_preference")
    print(f"   Stored preference: {pref}")

    # 4. Mock Tool Router
    router = registry.get("mock_tool_router", namespace="plugin")
    print("\n4. Mock Tool Router:")
    tools = router.get_available_tools()
    print(f"   Available tools: {tools}")

    result = await router.acall_tool("calculate", {"expression": "42 * 2"})
    print(f"   Calculation result: {result}")

    print("\n✅ Mock adapters demonstration complete!")


async def demo_workflow_with_mocks():
    """Demonstrate a complete workflow using mock adapters."""
    print("\n" + "=" * 60)
    print("Complete Workflow with Mock Adapters")
    print("=" * 60)

    # This would typically be done in a DAG node
    from hexai.core.application.nodes import FunctionNode

    # Create a node that uses multiple mock adapters
    async def process_customer_query(input_data, **kwargs):
        """Process a customer query using mock adapters."""
        from hexai.core.context import get_port

        llm = get_port("llm")
        db = get_port("database")
        memory = get_port("memory")
        tools = get_port("tool_router")

        # Check memory for cached response
        cached = await memory.aget(f"query_{input_data['query_id']}")
        if cached:
            return {"source": "cache", "response": cached}

        # Query database for customer info
        # Note: Using parameterized query to avoid SQL injection
        # (mock adapter doesn't actually execute SQL)
        customer_data = await db.aexecute_query(
            "SELECT * FROM customers WHERE id = ?", {"id": input_data["customer_id"]}
        )

        # Use tool to get additional info
        weather = await tools.acall_tool(
            "get_weather", {"location": input_data.get("location", "Unknown")}
        )

        # Generate response with LLM
        from hexai.core.ports.llm import Message

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

    # Get mock adapters directly from their modules for creating new instances
    from hexai.adapters.memory import InMemoryMemory
    from hexai.adapters.mock.mock_database import MockDatabaseAdapter
    from hexai.adapters.mock.mock_llm import MockLLM
    from hexai.adapters.mock.mock_tool_router import MockToolRouter

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

    print("\n📥 Input:")
    print(f"   {input_data}")

    # Use ExecutionContext to make ports available to the function
    from hexai.core.context import ExecutionContext

    async with ExecutionContext(ports=ports):
        result = await node.fn(input_data)

    print("\n📤 Output:")
    print(f"   Source: {result['source']}")
    print(f"   Response: {result['response']}")
    print(f"   Weather: {result['weather']}")

    # Run again to test caching
    print("\n🔄 Running again (should use cache):")
    async with ExecutionContext(ports=ports):
        result2 = await node.fn(input_data)
    print(f"   Source: {result2['source']}")
    print(f"   Response: {result2['response']}")


async def main():
    """Run all demonstrations."""
    await demo_mock_adapters()
    await demo_workflow_with_mocks()


if __name__ == "__main__":
    asyncio.run(main())
