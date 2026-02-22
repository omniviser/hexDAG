"""
Simple Pandas Transform Example

This example demonstrates the PandasTransformNode without artifact storage
"""

import asyncio
import sys
from pathlib import Path

import pandas as pd

# Add hexdag and plugin to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))  # hexdag root
sys.path.insert(0, str(Path(__file__).parent.parent))  # plugin root

# Import plugin to register components
from hexdag.kernel.orchestration.orchestrator import Orchestrator
from hexdag.kernel.pipeline_builder.yaml_builder import YamlPipelineBuilder
from hexdag.stdlib.adapters.memory.in_memory_memory import InMemoryMemory

import hexdag_etl  # noqa: F401


def create_sample_data():
    """Create a simple DataFrame."""
    return pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003", "C004", "C005"],
            "name": ["Alice", "Bob", "Carol", "David", "Emma"],
            "amount": [150.0, 299.99, 150.0, 29.99, 299.99],
            "category": ["A", "B", "A", "C", "B"],
        }
    )


def main():
    """Run the example."""
    print("=" * 80)
    print("hexDAG ETL Example: Simple Pandas Transform")
    print("=" * 80)

    # Create sample data
    df = create_sample_data()
    print("\n✓ Created sample DataFrame:")
    print(df)

    # Define pipeline YAML
    pipeline_yaml = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: pandas-transform-demo
  description: Demonstrate pandas multi-operation transforms
spec:
  nodes:
    # Create DataFrame
    - kind: function_node
      metadata:
        name: create_data
      spec:
        fn: "pandas.DataFrame"
        input_schema:
          data: dict
        output_schema:
          output: dataframe
      dependencies: []

    # Transform with pandas operations
    - kind: user:pandas_transform_node
      metadata:
        name: transform_data
      spec:
        operations:
          # Sort by amount
          - type: transform
            method: pandas.DataFrame.sort_values
            kwargs:
              by: amount
              ascending: false

          # Add a new column
          - type: transform
            method: pandas.DataFrame.assign
            kwargs:
              amount_doubled: "{{ lambda df: df['amount'] * 2 }}"

          # Get top 3 rows
          - type: transform
            method: pandas.DataFrame.head
            args:
              - 3
        dependencies: [create_data]

    # Display results
    - kind: function_node
      metadata:
        name: display_results
      spec:
        fn: "builtins.print"
        input_schema:
          output: dict
      dependencies: [transform_data]
"""

    print("\n✓ Building pipeline from YAML...")
    builder = YamlPipelineBuilder()
    graph, config = builder.build_from_yaml_string(pipeline_yaml)

    print(f"✓ Pipeline built with {len(graph.nodes)} nodes")

    # Prepare inputs
    inputs = {
        "create_data": {
            "data": {
                "customer_id": ["C001", "C002", "C003", "C004", "C005"],
                "name": ["Alice", "Bob", "Carol", "David", "Emma"],
                "amount": [150.0, 299.99, 150.0, 29.99, 299.99],
                "category": ["A", "B", "A", "C", "B"],
            }
        }
    }

    # Execute pipeline
    print("\n✓ Executing pipeline...")
    orchestrator = Orchestrator(ports={"memory": InMemoryMemory()})

    try:
        result = asyncio.run(orchestrator.run(graph, initial_input=inputs))

        print("\n" + "=" * 80)
        print("✓ Pipeline execution completed successfully!")
        print("=" * 80)

        # Show transformed output
        transformed_df = result["transform_data"]["output"]
        print("\n📊 Transformed Data:")
        print(transformed_df)

    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
