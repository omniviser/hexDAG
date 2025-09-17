#!/usr/bin/env python3
"""Example demonstrating lazy loading of optional dependencies.

This example shows how HexDAG handles optional dependencies gracefully,
only importing them when actually needed.
"""


def test_core_functionality():
    """Test that core functionality works without optional dependencies."""
    print("=" * 60)
    print("Testing Core Functionality (No Optional Dependencies)")
    print("=" * 60)

    # Core imports should always work
    from hexai import (
        DirectedGraph,
        NodeSpec,
        registry,
    )

    print("✅ Core imports successful")

    # Create a simple DAG
    graph = DirectedGraph()
    graph.add_node(NodeSpec(id="start", fn=lambda x: x))
    print("✅ Can create DirectedGraph and NodeSpec")

    # Check registry
    components = registry.list_components()
    print(f"✅ Registry has {len(components)} components")


def test_visualization_lazy():
    """Test lazy loading of visualization module."""
    print("\n" + "=" * 60)
    print("Testing Visualization Lazy Loading")
    print("=" * 60)

    # Check if graphviz is available without importing
    from hexai.visualization import GRAPHVIZ_AVAILABLE

    print(f"Graphviz available: {GRAPHVIZ_AVAILABLE}")

    if GRAPHVIZ_AVAILABLE:
        # Only import when needed
        from hexai.visualization import DAGVisualizer

        print("✅ DAGVisualizer imported successfully")

        # Create a visualizer
        from hexai import DirectedGraph, NodeSpec

        graph = DirectedGraph()
        graph.add_node(NodeSpec(id="test", fn=lambda x: x))

        DAGVisualizer(graph)
        print("✅ DAGVisualizer created successfully")
    else:
        print("⚠️  Graphviz not installed - visualization features unavailable")
        print("   Install with: pip install hexdag[viz]")

        # Try to import anyway to show the error
        try:
            from hexai.visualization import DAGVisualizer

            print("❌ This shouldn't happen!")
        except ImportError as e:
            print(f"✅ Import error as expected: {e}")


def test_yaml_lazy():
    """Test lazy loading of YAML functionality."""
    print("\n" + "=" * 60)
    print("Testing YAML/Agent Factory Lazy Loading")
    print("=" * 60)

    # Check if YAML is available without importing
    from hexai.agent_factory import YAML_AVAILABLE

    print(f"YAML available: {YAML_AVAILABLE}")

    if YAML_AVAILABLE:
        # Only import when needed
        from hexai.agent_factory import YamlPipelineBuilder

        print("✅ YamlPipelineBuilder imported successfully")

        # The builder would work here
        print("✅ YAML support is available")
    else:
        print("⚠️  PyYAML not installed - YAML features unavailable")
        print("   Install with: pip install hexdag[cli]")

        # Try to import anyway to show the error
        try:
            from hexai.agent_factory import YamlPipelineBuilder

            print("❌ This shouldn't happen!")
        except ImportError as e:
            print(f"✅ Import error as expected: {e}")


def test_lazy_getattr():
    """Test that __getattr__ lazy loading works."""
    print("\n" + "=" * 60)
    print("Testing __getattr__ Lazy Loading")
    print("=" * 60)

    # Import hexai module
    import hexai

    print("✅ hexai module imported")

    # Try to access visualization (may or may not be available)
    try:
        # This triggers __getattr__ in hexai/__init__.py
        visualizer_class = hexai.DAGVisualizer
        print(f"✅ Got DAGVisualizer class: {visualizer_class}")
    except ImportError as e:
        print(f"⚠️  DAGVisualizer not available: {e}")

    # Mock adapters should always work (they're included in core)
    try:
        mock_llm = hexai.MockLLM
        print(f"✅ Got MockLLM class: {mock_llm}")
    except ImportError as e:
        print(f"❌ MockLLM should be available: {e}")


def test_performance():
    """Test that lazy loading improves import performance."""
    print("\n" + "=" * 60)
    print("Testing Import Performance")
    print("=" * 60)

    import time

    # Time core import
    start = time.time()

    core_time = time.time() - start
    print(f"Core domain import time: {core_time:.4f}s")

    # Time optional import (only if available)
    try:
        start = time.time()
        from hexai.visualization import GRAPHVIZ_AVAILABLE

        check_time = time.time() - start
        print(f"Availability check time: {check_time:.4f}s")

        if GRAPHVIZ_AVAILABLE:
            start = time.time()
            from hexai.visualization import DAGVisualizer

            viz_time = time.time() - start
            print(f"Visualization import time: {viz_time:.4f}s")
    except ImportError:
        print("Visualization not available for timing")


def main():
    """Run all lazy loading tests."""
    print("🚀 HexDAG Lazy Loading Demo\n")

    # Test each aspect
    test_core_functionality()
    test_visualization_lazy()
    test_yaml_lazy()
    test_lazy_getattr()
    test_performance()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
Lazy loading benefits:
1. ⚡ Faster imports - only load what you need
2. 📦 Smaller deployments - optional deps stay optional
3. 🛡️ Better security - fewer dependencies = smaller attack surface
4. 🎯 Clear errors - helpful messages when features unavailable
5. 🔧 Graceful degradation - core works without optional features

Install options:
- Minimal: pip install hexdag
- With CLI: pip install hexdag[cli]
- With Viz: pip install hexdag[viz]
- Everything: pip install hexdag[all]

Or with uv:
- Minimal: uv pip install hexdag
- With CLI: uv pip install hexdag[cli]
- With Viz: uv pip install hexdag[viz]
- Everything: uv pip install hexdag[all]
""")


if __name__ == "__main__":
    main()
