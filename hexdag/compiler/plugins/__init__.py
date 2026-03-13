"""Entity plugins for the YAML pipeline builder.

These plugins build specific entity types from YAML configuration:
- AdapterDefinitionPlugin: Process Adapter definitions (kind: Adapter)
- ConfigDefinitionPlugin: Process Config definitions (kind: Config)
- MacroDefinitionPlugin: Process Macro definitions (kind: Macro)
- MacroEntityPlugin: Expand macro invocations into subgraphs
- MiddlewareDefinitionPlugin: Process Middleware definitions (kind: Middleware)
- NodeEntityPlugin: Build regular nodes (llm_node, function_node, etc.)
"""

from hexdag.compiler.plugins.adapter_definition import AdapterDefinitionPlugin
from hexdag.compiler.plugins.config_definition import ConfigDefinitionPlugin
from hexdag.compiler.plugins.macro_definition import MacroDefinitionPlugin
from hexdag.compiler.plugins.macro_entity import MacroEntityPlugin
from hexdag.compiler.plugins.middleware_definition import MiddlewareDefinitionPlugin
from hexdag.compiler.plugins.node_entity import NodeEntityPlugin

__all__ = [
    "AdapterDefinitionPlugin",
    "ConfigDefinitionPlugin",
    "MacroDefinitionPlugin",
    "MacroEntityPlugin",
    "MiddlewareDefinitionPlugin",
    "NodeEntityPlugin",
]
