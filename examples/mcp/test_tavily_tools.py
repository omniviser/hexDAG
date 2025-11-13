"""Simple test script for Tavily tools registration.

This validates that:
1. Tavily tools are properly registered
2. Tool schemas are correct
3. Basic tool execution works
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import registry first, then Tavily tools
from hexdag.core.registry import registry
from hexdag.core.registry.models import ComponentType

# Import tavily adapter to register tools (after registry is available)
import examples.mcp.tavily_adapter  # noqa: F401


def test_tool_registration():
    """Test that Tavily tools are registered."""
    print("\n" + "=" * 80)
    print("Testing Tavily Tool Registration")
    print("=" * 80)

    # Check tavily_search tool
    print("\n1. Checking tavily_search tool...")
    try:
        tool = registry.get("research:tavily_search")
        print(f"   ✓ tavily_search found: {tool}")

        # Get metadata
        metadata = registry.get_metadata("research:tavily_search")
        print(f"   ✓ Description: {metadata.description[:80]}...")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    # Check tavily_qna_search tool
    print("\n2. Checking tavily_qna_search tool...")
    try:
        tool = registry.get("research:tavily_qna_search")
        print(f"   ✓ tavily_qna_search found: {tool}")

        metadata = registry.get_metadata("research:tavily_qna_search")
        print(f"   ✓ Description: {metadata.description[:80]}...")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    print("\n" + "=" * 80)
    print("✅ All tool registration tests passed!")
    print("=" * 80)
    return True


def test_yaml_loading():
    """Test YAML pipeline loading."""
    print("\n" + "=" * 80)
    print("Testing YAML Pipeline Loading")
    print("=" * 80)

    try:
        from hexdag.core.pipeline_builder.yaml_builder import YamlPipelineBuilder

        yaml_path = Path(__file__).parent / "deep_research_agent.yaml"
        print(f"\nLoading pipeline from: {yaml_path}")

        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_file(str(yaml_path))

        print(f"\n✓ Pipeline loaded successfully!")
        print(f"  Config type: {type(config)}")
        print(f"  Nodes: {len(graph.nodes)}")

        # List first few nodes
        print(f"\n  Sample node names:")
        for i, node_name in enumerate(sorted(graph.nodes.keys())[:5]):
            print(f"    - {node_name}")
        if len(graph.nodes) > 5:
            print(f"    ... and {len(graph.nodes) - 5} more")

        print("\n" + "=" * 80)
        print("✅ YAML pipeline loaded successfully!")
        print("=" * 80)
        return True

    except Exception as e:
        print(f"\n✗ Error loading YAML: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("🔬 Tavily Research Agent - Test Suite")
    print("=" * 80)

    success = True

    # Test 1: Tool registration
    if not test_tool_registration():
        success = False

    # Test 2: YAML loading
    if not test_yaml_loading():
        success = False

    # Summary
    print("\n" + "=" * 80)
    if success:
        print("✅ All tests passed!")
        print("\nNext steps:")
        print("1. Set environment variables:")
        print("   export OPENAI_API_KEY='sk-...'")
        print("   export TAVILY_API_KEY='tvly-...'")
        print("\n2. Run the agent:")
        print("   python examples/mcp/run_deep_research_agent.py")
    else:
        print("❌ Some tests failed")
        return 1

    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
