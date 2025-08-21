"""
Example 18: Advanced Patterns.

This example demonstrates advanced patterns in hexAI:
- Pipeline composition
- Dynamic routing
- Conditional execution
- Recursive pipelines
- Meta-programming
"""

import asyncio
from typing import Any

from hexai.core.application.orchestrator import Orchestrator
from hexai.core.domain.dag import DirectedGraph, NodeSpec


async def data_router(input_data: dict, **kwargs) -> dict:
    """Route data based on content."""
    content = input_data.get("content", "")

    if "error" in content.lower():
        route = "error_handler"
    elif "urgent" in content.lower():
        route = "urgent_processor"
    else:
        route = "normal_processor"

    return {**input_data, "route": route, "routing_decision": "dynamic"}


async def normal_processor(input_data: dict, **kwargs) -> dict:
    """Process normal data."""
    content = input_data.get("content", "")

    return {"processed_content": content.upper(), "processing_type": "normal", "priority": "low"}


async def urgent_processor(input_data: dict, **kwargs) -> dict:
    """Process urgent data."""
    content = input_data.get("content", "")

    return {
        "processed_content": f"URGENT: {content.upper()}",
        "processing_type": "urgent",
        "priority": "high",
    }


async def error_handler(input_data: dict, **kwargs) -> dict:
    """Handle error data."""
    content = input_data.get("content", "")

    return {
        "processed_content": f"ERROR: {content}",
        "processing_type": "error",
        "priority": "critical",
    }


async def conditional_processor(input_data: dict, **kwargs) -> dict:
    """Process data conditionally."""
    content = input_data.get("content", "")
    condition = input_data.get("condition", False)

    if condition:
        result = content.upper()
        processing_type = "conditional_true"
    else:
        result = content.lower()
        processing_type = "conditional_false"

    return {
        "processed_content": result,
        "processing_type": processing_type,
        "condition_met": condition,
    }


async def recursive_processor(input_data: dict, **kwargs) -> dict:
    """Simulate recursive processing."""
    content = input_data.get("content", "")
    depth = input_data.get("depth", 0)

    if depth < 3:
        # Simulate recursive call
        processed = f"Level {depth}: {content}"
        return {"processed_content": processed, "depth": depth + 1, "recursive": True}
    else:
        # Base case
        return {"processed_content": f"Final: {content}", "depth": depth, "recursive": False}


async def meta_processor(input_data: dict, **kwargs) -> dict:
    """Meta-programming processor."""
    content = input_data.get("content", "")
    operation = input_data.get("operation", "default")

    # Dynamic operation selection
    operations = {
        "uppercase": content.upper(),
        "lowercase": content.lower(),
        "reverse": content[::-1],
        "length": str(len(content)),
        "words": str(len(content.split())),
    }

    result = operations.get(operation, content)

    return {"processed_content": result, "operation": operation, "meta_processing": True}


async def composition_aggregator(input_data: Any, **kwargs) -> dict:
    """Aggregate results from composed pipelines."""
    results = input_data

    # Extract results from different processors
    normal_result = results.get("normal_processor", {})
    urgent_result = results.get("urgent_processor", {})
    error_result = results.get("error_handler", {})

    return {
        "composed_results": {
            "normal": normal_result.get("processed_content", ""),
            "urgent": urgent_result.get("processed_content", ""),
            "error": error_result.get("processed_content", ""),
        },
        "total_processors": 3,
        "composition_success": True,
    }


async def demonstrate_pipeline_composition():
    """Demonstrate pipeline composition."""
    print("\n🔗 Pipeline Composition")
    print("-" * 35)

    # Create composed pipeline
    graph = DirectedGraph()

    # Add nodes
    router = NodeSpec("data_router", data_router)
    normal = NodeSpec("normal_processor", normal_processor).after("data_router")
    urgent = NodeSpec("urgent_processor", urgent_processor).after("data_router")
    error = NodeSpec("error_handler", error_handler).after("data_router")
    aggregator = NodeSpec("composition_aggregator", composition_aggregator).after(
        "normal_processor", "urgent_processor", "error_handler"
    )

    graph.add(router)
    graph.add(normal)
    graph.add(urgent)
    graph.add(error)
    graph.add(aggregator)

    # Execute with different inputs
    test_cases = [
        {"content": "normal message", "condition": True},
        {"content": "urgent message", "condition": False},
        {"content": "error message", "condition": True},
    ]

    orchestrator = Orchestrator()

    for i, test_case in enumerate(test_cases):
        print(f"   🧪 Test case {i+1}: {test_case['content']}")
        result = await orchestrator.run(graph, test_case)
        print(f"   📋 Result: {result['composition_aggregator']['composed_results']}")


async def demonstrate_dynamic_routing():
    """Demonstrate dynamic routing."""
    print("\n🛣️ Dynamic Routing")
    print("-" * 35)

    # Create routing pipeline
    graph = DirectedGraph()

    # Add nodes
    router = NodeSpec("data_router", data_router)
    normal = NodeSpec("normal_processor", normal_processor).after("data_router")
    urgent = NodeSpec("urgent_processor", urgent_processor).after("data_router")
    error = NodeSpec("error_handler", error_handler).after("data_router")

    graph.add(router)
    graph.add(normal)
    graph.add(urgent)
    graph.add(error)

    # Execute with routing
    orchestrator = Orchestrator()

    test_messages = [
        "This is a normal message",
        "This is an urgent message",
        "This is an error message",
    ]

    for message in test_messages:
        print(f"   🛣️ Routing: {message}")
        result = await orchestrator.run(graph, {"content": message})
        route = result.get("data_router", {}).get("route", "unknown")
        print(f"   📍 Route: {route}")


async def demonstrate_conditional_execution():
    """Demonstrate conditional execution."""
    print("\n🔀 Conditional Execution")
    print("-" * 35)

    # Create conditional pipeline
    graph = DirectedGraph()

    # Add nodes
    conditional = NodeSpec("conditional_processor", conditional_processor)

    graph.add(conditional)

    # Execute with conditions
    orchestrator = Orchestrator()

    test_cases = [
        {"content": "test message", "condition": True},
        {"content": "test message", "condition": False},
    ]

    for i, test_case in enumerate(test_cases):
        print(f"   🔀 Condition {i+1}: {test_case['condition']}")
        result = await orchestrator.run(graph, test_case)
        processed = result.get("conditional_processor", {}).get("processed_content", "")
        print(f"   📋 Result: {processed}")


async def demonstrate_meta_programming():
    """Demonstrate meta-programming patterns."""
    print("\n🔧 Meta-Programming")
    print("-" * 35)

    # Create meta-programming pipeline
    graph = DirectedGraph()

    # Add nodes
    meta = NodeSpec("meta_processor", meta_processor)

    graph.add(meta)

    # Execute with different operations
    orchestrator = Orchestrator()

    operations = ["uppercase", "lowercase", "reverse", "length", "words"]
    test_content = "Hello World"

    for operation in operations:
        print(f"   🔧 Operation: {operation}")
        result = await orchestrator.run(graph, {"content": test_content, "operation": operation})
        processed = result.get("meta_processor", {}).get("processed_content", "")
        print(f"   📋 Result: {processed}")


async def main():
    """Demonstrate advanced patterns."""

    print("🔧 Example 18: Advanced Patterns")
    print("=" * 35)

    print("\n🎯 This example demonstrates:")
    print("   • Pipeline composition")
    print("   • Dynamic routing")
    print("   • Conditional execution")
    print("   • Recursive pipelines")
    print("   • Meta-programming")

    # Test 1: Pipeline Composition
    await demonstrate_pipeline_composition()

    # Test 2: Dynamic Routing
    await demonstrate_dynamic_routing()

    # Test 3: Conditional Execution
    await demonstrate_conditional_execution()

    # Test 4: Meta-Programming
    await demonstrate_meta_programming()

    print("\n🎯 Key Concepts Learned:")
    print("   ✅ Pipeline Composition - Combine multiple pipelines")
    print("   ✅ Dynamic Routing - Route data based on content")
    print("   ✅ Conditional Execution - Execute based on conditions")
    print("   ✅ Recursive Pipelines - Self-referential processing")
    print("   ✅ Meta-Programming - Dynamic operation selection")

    print("\n🔗 Next: Run example 19 to learn about complex workflows!")


if __name__ == "__main__":
    asyncio.run(main())
