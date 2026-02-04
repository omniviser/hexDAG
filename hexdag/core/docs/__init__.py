"""Documentation generation framework for hexDAG.

This module provides tools to extract documentation from code artifacts
(decorators, signatures, docstrings) and generate up-to-date documentation
for the MCP server and other consumers.
"""

from hexdag.core.docs.extractors import DocExtractor
from hexdag.core.docs.generators import GuideGenerator
from hexdag.core.docs.models import (
    AdapterDoc,
    ComponentDoc,
    NodeDoc,
    ParameterDoc,
    ToolDoc,
)

__all__ = [
    "AdapterDoc",
    "ComponentDoc",
    "DocExtractor",
    "GuideGenerator",
    "NodeDoc",
    "ParameterDoc",
    "ToolDoc",
]
