#!/usr/bin/env python3
"""
🎨 Example 27: Operator Overloading Showcase

This example demonstrates all the operator simplifications available in hexDAG.
See docs/OPERATOR_SIMPLIFICATIONS.md for detailed comparisons.

Run: python examples/27_operator_showcase.py
"""

import asyncio

from hexdag.core.domain.dag import DirectedGraph, NodeSpec
from hexdag.core.orchestration.orchestrator import Orchestrator
from hexdag.core.ports_builder import PortsBuilder


# Simple processing functions
async def load_data(x: int) -> int:
    """Load initial data."""
    return x * 2


async def process_data(x: int) -> int:
    """Process the data."""
    return x + 10


async def validate_data(x: int) -> int:
    """Validate the data."""
    return x - 5


async def save_data(x: int) -> dict:
    """Save the final result."""
    return {"result": x, "status": "saved"}


async def main():
    """Demonstrate all operator simplifications."""
    print("🎨 Example 27: Operator Overloading Showcase")
    print("=" * 60)

    # ============================================================================
    # Part 1: DirectedGraph Operators
    # ============================================================================
    print("\n" + "=" * 60)
    print("PART 1: DirectedGraph Operators")
    print("=" * 60)

    # Operator: += (add single node)
    print("\n✨ Operator: graph += node")
    print("-" * 60)
    graph1 = DirectedGraph()
    loader = NodeSpec("loader", load_data)
    graph1 += loader
    print(f"   Added node: {loader.name}")
    print(f"   ✅ Graph now has {len(graph1)} node(s)")

    # Operator: += (add multiple nodes)
    print("\n✨ Operator: graph += [node1, node2, node3]")
    print("-" * 60)
    graph2 = DirectedGraph()
    nodes = [
        NodeSpec("load", load_data),
        NodeSpec("process", process_data),
        NodeSpec("save", save_data),
    ]
    graph2 += nodes
    print(f"   Added {len(nodes)} nodes in one operation")
    print(f"   ✅ Graph now has {len(graph2)} node(s)")

    # Operator: in (membership check)
    print("\n✨ Operator: 'node_name' in graph")
    print("-" * 60)
    if "load" in graph2:
        print("   ✅ Found 'load' node in graph")
    if "nonexistent" not in graph2:
        print("   ✅ 'nonexistent' node not in graph")

    # Operator: iter (iteration)
    print("\n✨ Operator: for node in graph")
    print("-" * 60)
    print("   Iterating over nodes:")
    for node in graph2:
        print(f"      • {node.name}")

    # Operator: len (size)
    print("\n✨ Operator: len(graph)")
    print("-" * 60)
    print(f"   Graph size: {len(graph2)} nodes")

    # ============================================================================
    # Part 2: Pipeline Operators (Visual Dependencies)
    # ============================================================================
    print("\n" + "=" * 60)
    print("PART 2: Pipeline Operators (Visual Dependencies)")
    print("=" * 60)

    # Operator: >> (visual dependency)
    print("\n✨ Operator: node_a >> node_b (visual pipeline)")
    print("-" * 60)
    print("   Creating visual pipeline: load >> process >> validate >> save")

    graph3 = DirectedGraph()

    # Create nodes
    load = NodeSpec("load", load_data)
    process = NodeSpec("process", process_data)
    validate = NodeSpec("validate", validate_data)
    save = NodeSpec("save", save_data)

    # Build pipeline with >> operator
    graph3 += load
    dep1 = load >> process
    graph3 += dep1
    dep2 = dep1 >> validate
    graph3 += dep2
    graph3 += dep2 >> save

    print("\n   Pipeline structure:")
    print("      load → process → validate → save")
    print("\n   Dependencies:")
    for node in graph3:
        deps = graph3.get_dependencies(node.name)
        if deps:
            print(f"      {node.name} depends on: {list(deps)}")
        else:
            print(f"      {node.name} has no dependencies (entry node)")

    # Validate and show execution order
    graph3.validate()
    waves = graph3.waves()
    print(f"\n   Execution order ({len(waves)} waves):")
    for i, wave in enumerate(waves, 1):
        print(f"      Wave {i}: {wave}")

    # ============================================================================
    # Part 3: Graph Merging
    # ============================================================================
    print("\n" + "=" * 60)
    print("PART 3: Graph Merging Operator")
    print("=" * 60)

    # Operator: |= (merge graphs)
    print("\n✨ Operator: graph |= subgraph")
    print("-" * 60)

    # Create two separate graphs
    pipeline_a = DirectedGraph()
    pipeline_a += NodeSpec("step1", load_data)
    pipeline_a += NodeSpec("step2", process_data)

    pipeline_b = DirectedGraph()
    pipeline_b += NodeSpec("step3", validate_data)
    pipeline_b += NodeSpec("step4", save_data)

    print(f"   Pipeline A has {len(pipeline_a)} nodes: {list(pipeline_a.nodes.keys())}")
    print(f"   Pipeline B has {len(pipeline_b)} nodes: {list(pipeline_b.nodes.keys())}")

    # Merge pipeline_b into pipeline_a
    pipeline_a |= pipeline_b

    print("\n   After merge: pipeline_a |= pipeline_b")
    print(f"   ✅ Combined pipeline has {len(pipeline_a)} nodes: {list(pipeline_a.nodes.keys())}")

    # ============================================================================
    # Part 4: Fluent Chaining
    # ============================================================================
    print("\n" + "=" * 60)
    print("PART 4: Fluent Chaining Operator")
    print("=" * 60)

    # Operator: << (fluent chaining)
    print("\n✨ Operator: graph << node1 << node2 << node3")
    print("-" * 60)

    graph4 = DirectedGraph()
    graph4 << NodeSpec("a", load_data) << NodeSpec("b", process_data) << NodeSpec("c", save_data)

    print(f"   Added {len(graph4)} nodes with fluent chaining")
    print(f"   Nodes: {list(graph4.nodes.keys())}")

    # ============================================================================
    # Part 5: PortsBuilder Operators
    # ============================================================================
    print("\n" + "=" * 60)
    print("PART 5: PortsBuilder Dict-Like Operators")
    print("=" * 60)

    # Operator: [] (dict-like access)
    print("\n✨ Operator: ports['key'] = value")
    print("-" * 60)

    ports = PortsBuilder()

    # Simulate some adapters
    class MockLLM:
        def __init__(self, name):
            self.name = name

    # Global level
    ports["llm"] = MockLLM("default-llm")
    print("   Set global port: ports['llm'] = default-llm")

    # Type level
    ports["agent", "llm"] = MockLLM("agent-llm")
    print("   Set type-level port: ports['agent', 'llm'] = agent-llm")

    # Node level
    ports["researcher", "agent", "llm"] = MockLLM("researcher-llm")
    print("   Set node-level port: ports['researcher', 'agent', 'llm'] = researcher-llm")

    # Demonstrate hierarchical resolution
    print("\n   Hierarchical resolution:")
    print(f"      Global level: {ports['llm'].name}")
    print(f"      Type level: {ports['agent', 'llm'].name}")
    print(f"      Node level: {ports['researcher', 'agent', 'llm'].name}")

    # Operator: in (membership)
    print("\n✨ Operator: 'key' in ports")
    print("-" * 60)
    if "llm" in ports:
        print("   ✅ 'llm' port is configured")

    # ============================================================================
    # Part 6: Real Execution
    # ============================================================================
    print("\n" + "=" * 60)
    print("PART 6: Execute Pipeline Built With Operators")
    print("=" * 60)

    print("\n🚀 Executing the visual pipeline from Part 2...")
    orchestrator = Orchestrator()
    results = await orchestrator.run(graph3, 5)

    print("\n📊 Results:")
    for node_name, result in results.items():
        print(f"   {node_name}: {result}")

    # ============================================================================
    # Summary
    # ============================================================================
    print("\n" + "=" * 60)
    print("SUMMARY: All Operators Demonstrated")
    print("=" * 60)

    print("\n✅ DirectedGraph Operators:")
    print("   • graph += node          → Add single node")
    print("   • graph += [n1, n2]      → Add multiple nodes")
    print("   • 'name' in graph        → Check if node exists")
    print("   • for node in graph      → Iterate over nodes")
    print("   • len(graph)             → Get node count")

    print("\n✅ Pipeline Operators:")
    print("   • node_a >> node_b       → Visual dependency (b depends on a)")
    print("   • graph |= subgraph      → Merge graphs")
    print("   • graph << node          → Fluent chaining")

    print("\n✅ PortsBuilder Operators:")
    print("   • ports[k] = v           → Set global port")
    print("   • ports[type, k] = v     → Set type-level port")
    print("   • ports[node, type, k]   → Get node-level port (hierarchical)")
    print("   • 'key' in ports         → Check if port exists")

    print("\n🎯 Key Benefits:")
    print("   ✅ 40-90% fewer lines of code")
    print("   ✅ More Pythonic and intuitive")
    print("   ✅ Visual representation of data flow")
    print("   ✅ Standard Python idioms")
    print("   ✅ Reduced cognitive load")

    print("\n📚 See docs/OPERATOR_SIMPLIFICATIONS.md for detailed comparisons!")


if __name__ == "__main__":
    asyncio.run(main())
