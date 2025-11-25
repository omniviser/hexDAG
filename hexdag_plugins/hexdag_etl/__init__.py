"""hexdag-etl: ETL infrastructure for hexDAG pipelines.

Provides:
- Multi-operation pandas transform nodes for data transformation
- API extraction with pagination and authentication (placeholder)
- SQL extraction and loading (placeholder implementations)

This plugin extends hexDAG with data pipeline capabilities for ETL workflows.
"""

from .nodes import PandasTransformNode

__version__ = "0.1.0"

__all__ = [
    "PandasTransformNode",
]
