"""Example: File-based ETL pipeline with CSV input and output.

This example demonstrates:
1. Reading CSV files with FileReaderNode
2. Writing results to CSV with FileWriterNode

For a complete example with transforms, use YAML pipelines where
template expressions ({{ }}) are properly resolved.
"""

import asyncio
import tempfile
from pathlib import Path

import pandas as pd


async def main() -> None:
    """Run a simple file-based ETL pipeline (read -> write)."""
    # Import after ensuring the module path is set
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

    from hexdag.core.domain.dag import DirectedGraph
    from hexdag.core.orchestration.orchestrator import Orchestrator

    # Import ETL nodes
    from hexdag_plugins.hexdag_etl.hexdag_etl.nodes.file_io import (
        FileReaderNode,
        FileWriterNode,
    )

    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create sample input CSV
        input_csv = tmpdir_path / "sales_data.csv"
        sample_data = pd.DataFrame(
            {
                "product": ["Widget A", "Widget B", "Widget C", "Widget A", "Widget B"],
                "quantity": [10, 5, 8, 15, 3],
                "price": [29.99, 49.99, 19.99, 29.99, 49.99],
                "region": ["North", "South", "North", "East", "West"],
            }
        )
        sample_data.to_csv(input_csv, index=False)
        print(f"Created input CSV: {input_csv}")
        print(f"Input data:\n{sample_data}\n")

        # Create output path (using CSV for portability - no pyarrow required)
        output_csv = tmpdir_path / "sales_copy.csv"

        # Build the pipeline graph
        graph = DirectedGraph()

        # Step 1: Read CSV file
        reader = FileReaderNode()
        read_spec = reader(
            name="read_sales",
            file_path=str(input_csv),
            format="csv",
        )
        graph.add(read_spec)

        # Step 2: Write to CSV (directly from reader output)
        writer = FileWriterNode()
        write_spec = writer(
            name="write_results",
            file_path=str(output_csv),
            format="csv",
            input_key="data",
            deps=["read_sales"],
        )
        graph.add(write_spec)

        # Execute the pipeline
        print("Executing ETL pipeline...")
        orchestrator = Orchestrator()
        results = await orchestrator.run(graph, {})

        # Display results
        print("\n=== Pipeline Results ===")
        for node_name, result in results.items():
            print(f"\n{node_name}:")
            if isinstance(result, dict):
                for key, value in result.items():
                    if key == "data" and isinstance(value, pd.DataFrame):
                        print(f"  {key}: DataFrame with {len(value)} rows")
                    elif key == "columns":
                        print(f"  {key}: {value}")
                    else:
                        print(f"  {key}: {value}")
            else:
                print(f"  {result}")

        # Verify output file
        if output_csv.exists():
            output_df = pd.read_csv(output_csv)
            print("\n=== Output CSV Contents ===")
            print(output_df)
            print(f"\nRows written: {len(output_df)}")
        else:
            print(f"\nWarning: Output file not created: {output_csv}")


if __name__ == "__main__":
    asyncio.run(main())
