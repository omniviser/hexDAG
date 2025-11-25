"""hexdag-etl: ETL infrastructure for hexDAG pipelines.

Provides:
- Multi-operation pandas transform nodes for data transformation
- API extraction with pagination and authentication (placeholder)
- SQL extraction and loading (placeholder implementations)

This plugin extends hexDAG with data pipeline capabilities for ETL workflows.
Note: Artifact storage adapter temporarily removed - pending architecture update.
"""

from .nodes import APIExtractNode, PandasTransformNode, SQLExtractNode, SQLLoadNode

__version__ = "0.1.0"

__all__ = [
    "PandasTransformNode",
    "APIExtractNode",
    "SQLExtractNode",
    "SQLLoadNode",
]
