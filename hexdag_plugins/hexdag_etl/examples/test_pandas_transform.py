"""Test pandas transform node directly without plugin registration."""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from hexdag.kernel.pipeline_builder.yaml_builder import YamlPipelineBuilder


def test_node_importable():
    """Test that node is importable."""
    print("Testing node import...")

    try:
        from hexdag_plugins.hexdag_etl.hexdag_etl import PandasTransformNode

        print(f"✓ PandasTransformNode: {PandasTransformNode}")

        # Test instantiation
        factory = PandasTransformNode()
        print(f"✓ Factory instance: {factory}")
        return True
    except Exception as e:
        print(f"❌ Error importing PandasTransformNode: {e}")
        return False


def test_yaml_parsing():
    """Test YAML pipeline parsing."""
    print("\nTesting YAML pipeline...")

    pipeline_yaml = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-pandas-transform
spec:
  nodes:
    - kind: etl:pandas_transform
      metadata:
        name: test_transform
      spec:
        operations:
          - type: transform
            method: pandas.DataFrame.head
            kwargs:
              n: 5
"""

    try:
        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_string(pipeline_yaml)
        print("\n✓ Pipeline built successfully!")
        print(f"✓ Nodes: {len(graph._graph.nodes())}")
        return True
    except Exception as e:
        print(f"\n❌ Error building pipeline: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 80)
    print("Testing Pandas Transform Node")
    print("=" * 80)

    reg_ok = test_node_importable()
    yaml_ok = test_yaml_parsing()

    if reg_ok and yaml_ok:
        print("\n" + "=" * 80)
        print("✓ All tests passed!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("❌ Some tests failed")
        print("=" * 80)
        sys.exit(1)
