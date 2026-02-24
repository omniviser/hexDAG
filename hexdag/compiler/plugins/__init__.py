"""Entity plugins for the YAML pipeline builder.

These plugins build specific entity types from YAML configuration:
- ConfigDefinitionPlugin: Process Config definitions (kind: Config)
- MacroDefinitionPlugin: Process Macro definitions (kind: Macro)
- MacroEntityPlugin: Expand macro invocations into subgraphs
- NodeEntityPlugin: Build regular nodes (llm_node, function_node, etc.)
"""

from hexdag.compiler.plugins.config_definition import ConfigDefinitionPlugin
from hexdag.compiler.plugins.macro_definition import MacroDefinitionPlugin
from hexdag.compiler.plugins.macro_entity import MacroEntityPlugin
from hexdag.compiler.plugins.node_entity import NodeEntityPlugin

__all__ = [
    "ConfigDefinitionPlugin",
    "MacroDefinitionPlugin",
    "MacroEntityPlugin",
    "NodeEntityPlugin",
]
