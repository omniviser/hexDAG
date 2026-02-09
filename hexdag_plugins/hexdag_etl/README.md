# hexDAG ETL Plugin

ETL (Extract, Transform, Load) infrastructure for hexDAG pipelines.

## Features

- **Artifact Storage**: Named storage slots for intermediate data between pipeline nodes
- **Pandas Transform**: Multi-operation DataFrame transformations with chaining support
- **API Extract**: REST API extraction with pagination, authentication, and rate limiting
- **SQL Operations**: Database extraction and loading (placeholder implementations)

## Installation

```bash
cd hexdag_plugins/hexdag_etl
pip install -e .
```

## Quick Start

### 1. Artifact Storage

Store intermediate data between pipeline nodes:

```python
from hexdag_plugins.hexdag_etl import LocalArtifactStore

# Create artifact store adapter
artifact_store = LocalArtifactStore(base_path="/tmp/etl_artifacts")

# Write artifact
await artifact_store.write(
    name="raw_customers",
    key="customers_2024_01_15",
    data=df
)

# Read artifact
df = await artifact_store.read(
    name="raw_customers",
    key="customers_2024_01_15"
)
```

### 2. Pandas Transform

Chain multiple pandas operations:

```yaml
- kind: etl:pandas_transform
  metadata:
    name: enrich_data
  spec:
    input_artifacts:
      - slot: raw_customers
        key: customers_v1
      - slot: raw_transactions
        key: transactions_v1
    operations:
      # Join DataFrames
      - type: transform
        method: pandas.merge
        args:
          - {{input_artifacts[0]}}
          - {{input_artifacts[1]}}
        kwargs:
          on: customer_id
          how: left

      # Rename columns
      - type: map
        columns:
          transaction_id: txn_id
          amount: total_amount

      # Add calculated column
      - type: transform
        method: pandas.DataFrame.assign
        kwargs:
          tier: |
            lambda df: pd.cut(df['amount'], bins=[0,100,500,inf])

    output_artifact:
      slot: enriched_data
      key: enriched_v1
```

### 3. API Extract

Extract data from REST APIs:

```yaml
- kind: etl:api_extract
  metadata:
    name: fetch_customers
  spec:
    endpoint: https://api.example.com/v1/customers
    method: GET
    params:
      limit: 100
      status: active
    pagination:
      type: cursor
      cursor_param: after
      cursor_path: meta.next_cursor
    auth:
      type: bearer
      token: ${API_TOKEN}
    output_artifact:
      slot: raw_customers
      key: customers_api
```

## Architecture

```
hexdag-etl/
├── hexdag_etl/
│   ├── adapters/
│   │   └── artifact.py          # LocalArtifactAdapter
│   ├── nodes/
│   │   ├── pandas_transform.py  # PandasTransformNode
│   │   ├── api_extract.py       # APIExtractNode
│   │   └── sql_extract_load.py  # SQLExtractNode, SQLLoadNode
│   └── ports/
│       └── artifact_storage.py  # ArtifactStorePort
├── examples/
├── tests/
└── pyproject.toml
```

## Components

### Adapters

- **LocalArtifactAdapter**: File-based artifact storage with compression and metadata

### Nodes

- **PandasTransformNode**: Multi-operation DataFrame transformations
- **APIExtractNode**: REST API extraction with pagination
- **SQLExtractNode**: Database extraction (placeholder)
- **SQLLoadNode**: Database loading (placeholder)

### Ports

- **ArtifactStorePort**: Interface for artifact storage adapters

## Examples

Run the example pipeline:

```bash
cd examples
python 01_simple_pandas_transform.py
```

## Testing

```bash
pytest tests/
```

## License

MIT
