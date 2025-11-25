"""
Example: Simple Pandas Transform Pipeline

This example demonstrates:
1. Reading CSV files
2. Applying multi-operation pandas transforms
3. Using artifact storage between nodes
4. Simple data cleaning and aggregation
"""

import asyncio
import sys
from pathlib import Path

import pandas as pd

# Add plugin to path and import to register components
sys.path.insert(0, str(Path(__file__).parent.parent))
from hexdag.core.memory import InMemoryMemory
from hexdag.core.orchestrator import Orchestrator
from hexdag.core.pipeline_builder.yaml_builder import YamlPipelineBuilder

import hexdag_etl  # noqa: F401 - import to register plugin components


def create_sample_data():
    """Create sample customer and transaction CSV files."""
    data_dir = Path("/tmp/hexdag_demo")
    data_dir.mkdir(exist_ok=True)

    # Sample customers
    customers_df = pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003", "C004", "C005"],
            "name": ["Alice", "Bob", "Carol", "David", "Emma"],
            "email": [
                "alice@example.com",
                "bob@example.com",
                "carol@example.com",
                "david@example.com",
                "emma@example.com",
            ],
            "country": ["USA", "UK", "USA", "Canada", "Australia"],
            "signup_date": pd.to_datetime(["2024-01-15", "2024-02-20", "2024-03-10", "2024-01-25", "2024-04-05"]),
        }
    )

    # Sample transactions
    transactions_df = pd.DataFrame(
        {
            "transaction_id": ["T001", "T002", "T003", "T004", "T005", "T006", "T007", "T008"],
            "customer_id": ["C001", "C001", "C002", "C003", "C004", "C005", "C001", "C003"],
            "product": ["Widget A", "Gadget B", "Widget A", "Book C", "Gadget B", "Widget A", "Book D", "Gadget C"],
            "category": [
                "Electronics",
                "Electronics",
                "Electronics",
                "Books",
                "Electronics",
                "Electronics",
                "Books",
                "Electronics",
            ],
            "amount": [150.00, 299.99, 150.00, 29.99, 299.99, 150.00, 19.99, 499.99],
            "date": pd.to_datetime(
                [
                    "2024-02-01",
                    "2024-03-15",
                    "2024-04-10",
                    "2024-02-20",
                    "2024-03-01",
                    "2024-04-15",
                    "2024-04-20",
                    "2024-05-01",
                ]
            ),
        }
    )

    # Save to CSV
    customers_file = data_dir / "customers.csv"
    transactions_file = data_dir / "transactions.csv"

    customers_df.to_csv(customers_file, index=False)
    transactions_df.to_csv(transactions_file, index=False)

    return str(customers_file), str(transactions_file)


def main():
    """Run the simple pandas transform example."""
    print("=" * 80)
    print("hexDAG ETL Example: Simple Pandas Transform")
    print("=" * 80)

    # Create sample data
    customers_file, transactions_file = create_sample_data()
    print("\n✓ Created sample data:")
    print(f"  - Customers: {customers_file}")
    print(f"  - Transactions: {transactions_file}")

    # Define pipeline YAML
    pipeline_yaml = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: simple-etl-demo
  description: Demonstrate pandas multi-operation transforms
spec:
  artifact_slots:
    - name: raw_customers
      type: dataframe

    - name: raw_transactions
      type: dataframe

    - name: enriched_customer_data
      type: dataframe

  ports:
    artifact_store:
      adapter: etl:local
      config:
        base_path: "/tmp/hexdag/artifacts"
        compress: true

  nodes:
    # Load customers CSV
    - kind: function_node
      metadata:
        name: load_customers
      spec:
        fn: pandas.read_csv
        input_schema:
          filepath_or_buffer: str
        output_schema:
          output: dataframe
        output_artifact:
          slot: raw_customers
          key: customers_v1

    # Load transactions CSV
    - kind: function_node
      metadata:
        name: load_transactions
      spec:
        fn: pandas.read_csv
        input_schema:
          filepath_or_buffer: str
        output_schema:
          output: dataframe
        kwargs:
          parse_dates: ["date"]
        output_artifact:
          slot: raw_transactions
          key: transactions_v1

    # Transform: Join and enrich data
    - kind: etl:pandas_transform
      metadata:
        name: enrich_customer_data
      spec:
        input_artifacts:
          - slot: raw_customers
            key: customers_v1
          - slot: raw_transactions
            key: transactions_v1
        operations:
          # Join customers with transaction summary
          - type: transform
            method: pandas.merge
            args:
              - {{input_artifacts[0]}}
              - |
                {{input_artifacts[1]}}
                .groupby('customer_id')
                .agg({
                  'transaction_id': 'count',
                  'amount': ['sum', 'mean']
                })
                .reset_index()
            kwargs:
              on: customer_id
              how: left
              suffixes: ["", "_txn"]

          # Rename columns
          - type: map
            columns:
              transaction_id: transaction_count
              ('amount', 'sum'): total_spend
              ('amount', 'mean'): avg_spend

          # Fill missing values
          - type: transform
            method: pandas.DataFrame.fillna
            kwargs:
              value:
                transaction_count: 0
                total_spend: 0.0
                avg_spend: 0.0

        output_artifact:
          slot: enriched_customer_data
          key: enriched_v1
      dependencies: [load_customers, load_transactions]

    # Display results
    - kind: function_node
      metadata:
        name: display_results
      spec:
        fn: "builtins.print"
        args:
          - "\n=== ETL Pipeline Complete ==="
          - "Customers loaded: {{output.load_customers.output.shape}}"
          - "Transactions loaded: {{output.load_transactions.output.shape}}"
          - "Enriched data: {{output.enrich_customer_data.records}} records"
        input_schema:
          output: dict
      dependencies: [enrich_customer_data]
"""

    print("\n✓ Building pipeline from YAML...")
    builder = YamlPipelineBuilder()
    graph, config = builder.build_from_string(pipeline_yaml)

    print(f"✓ Pipeline built with {len(graph._graph.nodes())} nodes")

    # Prepare inputs
    inputs = {
        "load_customers": {"filepath_or_buffer": customers_file},
        "load_transactions": {"filepath_or_buffer": transactions_file},
    }

    # Execute pipeline
    print("\n✓ Executing pipeline...")
    orchestrator = Orchestrator(
        memory=InMemoryMemory(),
        file_storage=None,  # Using artifact storage instead
    )

    try:
        asyncio.run(orchestrator.run_async(graph, inputs=inputs))

        print("\n" + "=" * 80)
        print("✓ Pipeline execution completed successfully!")
        print("=" * 80)

        # Show final artifact
        artifact_store = orchestrator.ports.get("artifact_store")
        if artifact_store:
            enriched_data = asyncio.run(artifact_store.read("enriched_customer_data", "enriched_v1"))
            print("\n📊 Final Enriched Data:")
            print(enriched_data.head())
            print(f"\nShape: {enriched_data.shape}")
            print(f"Columns: {list(enriched_data.columns)}")

    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
