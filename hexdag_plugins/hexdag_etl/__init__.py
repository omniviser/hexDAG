"""hexdag-etl: ETL infrastructure for hexDAG pipelines.

Provides file I/O and multi-operation pandas transform nodes for data transformation pipelines.

This plugin extends hexDAG with ETL capabilities:
- FileReaderNode: Read CSV, Parquet, JSON, Excel files
- FileWriterNode: Write data to various file formats
- PandasTransformNode: Chain pandas operations

Example Pipeline:
    - kind: etl:file_reader_node
      metadata:
        name: load_data
      spec:
        file_path: data/input.csv
        format: csv

    - kind: etl:pandas_transform_node
      metadata:
        name: transform
      spec:
        operations:
          - type: filter
            condition: "{{ df['value'] > 0 }}"
      dependencies: [load_data]

    - kind: etl:file_writer_node
      metadata:
        name: save_results
      spec:
        file_path: output/results.parquet
        format: parquet
      dependencies: [transform]
"""

# Re-export from the inner module (using relative import to inner hexdag_etl package)
from .hexdag_etl import (
    APIExtractNode,
    FileReaderNode,
    FileWriterNode,
    PandasTransformNode,
    SQLExtractNode,
    SQLLoadNode,
    __version__,
)

__all__ = [
    "__version__",
    "APIExtractNode",
    "FileReaderNode",
    "FileWriterNode",
    "PandasTransformNode",
    "SQLExtractNode",
    "SQLLoadNode",
]
