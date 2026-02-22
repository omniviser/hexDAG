"""Entity plugins for the YAML pipeline builder.

These plugins build specific entity types from YAML configuration:
- MacroDefinitionPlugin: Process Macro definitions (kind: Macro)
- MacroEntityPlugin: Expand macro invocations into subgraphs
- NodeEntityPlugin: Build regular nodes (llm_node, function_node, etc.)
"""

from hexdag.kernel.pipeline_builder.plugins.macro_definition import MacroDefinitionPlugin
from hexdag.kernel.pipeline_builder.plugins.macro_entity import MacroEntityPlugin
from hexdag.kernel.pipeline_builder.plugins.node_entity import NodeEntityPlugin

__all__ = [
    "MacroDefinitionPlugin",
    "MacroEntityPlugin",
    "NodeEntityPlugin",
]
