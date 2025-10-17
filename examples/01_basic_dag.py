#!/usr/bin/env python3
"""
🏁 Example 01: Basic DAG Construction

Your first hexAI pipeline! This example teaches:
- Creating a DirectedGraph
- Adding NodeSpec with simple functions
- Using pipeline operators (+=, >>) for clean syntax
- Running the orchestrator
- Understanding execution results

Run: python examples/01_basic_dag.py
"""

import asyncio

from hexdag.core.domain.dag import DirectedGraph, NodeSpec
from hexdag.core.orchestration.orchestrator import Orchestrator


async def step_one(input_data: str) -> dict:
    """First processing step - add a greeting."""
    return {"message": f"Hello, {input_data}!", "step": "greeting_complete"}


async def step_two(input_data: dict) -> dict:
    """Second processing step - add enthusiasm."""
    message = input_data.get("message", "")
    return {
        "message": f"{message} Welcome to hexAI!",
        "step": "enthusiasm_added",
        "original": input_data,
    }


async def step_three(input_data: dict) -> dict:
    """Final processing step - format result."""
    return {
        "final_message": input_data.get("message", ""),
        "processing_steps": ["greeting", "enthusiasm", "formatting"],
        "input_preserved": input_data.get("original", {}),
    }


async def main():
    """Demonstrate basic DAG construction and execution."""

    print("🏁 Example 01: Basic DAG Construction")
    print("=" * 50)

    # Step 1: Create a directed graph
    print("\n📊 Creating DirectedGraph...")
    graph = DirectedGraph()

    # Step 2: Add nodes with simple functions
    print("➕ Adding nodes...")

    # Create nodes
    node1 = NodeSpec("greeting", step_one)
    node2 = NodeSpec("enthusiasm", step_two)
    node3 = NodeSpec("formatting", step_three)

    # Add nodes with visual pipeline operator (>>)
    # The >> operator creates dependencies: node1 >> node2 means "node2 depends on node1"
    graph += node1
    graph += node1 >> node2  # node2 depends on node1
    graph += node2 >> node3  # node3 depends on node2

    print("   ✅ Added 'greeting' node")
    print("   ✅ Added 'enthusiasm' node (depends on greeting)")
    print("   ✅ Added 'formatting' node (depends on enthusiasm)")

    # Step 3: Validate the graph
    print("\n🔍 Validating graph structure...")
    try:
        graph.validate()
        print("   ✅ Graph validation passed!")
    except Exception as e:
        print(f"   ❌ Graph validation failed: {e}")
        return

    # Step 4: Show execution waves
    print("\n🌊 Execution waves:")
    waves = graph.waves()
    for i, wave in enumerate(waves, 1):
        print(f"   Wave {i}: {wave}")

    # Step 5: Create orchestrator and run
    print("\n🚀 Executing pipeline...")
    orchestrator = Orchestrator()

    # Run with input data
    input_name = "hexAI Learner"
    results = await orchestrator.run(graph, input_name)

    # Step 6: Show results
    print("\n📋 Execution Results:")
    for node_name, result in results.items():
        print(f"   {node_name}: {result}")

    print("\n🎯 Key Concepts Learned:")
    print("   ✅ DirectedGraph - Container for your workflow")
    print("   ✅ NodeSpec - Individual processing steps")
    print("   ✅ Pipeline Operators - >> creates dependencies, += adds to graph")
    print("   ✅ Orchestrator - Executes your DAG")
    print("   ✅ Results - Dictionary with node outputs")

    print("\n🔗 Next: Run example 02 to learn about complex dependencies!")


if __name__ == "__main__":
    asyncio.run(main())
