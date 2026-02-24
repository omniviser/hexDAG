"""Node factories for the hexdag framework.

All BaseNodeFactory subclasses in this package are auto-discovered
and registered for YAML pipeline validation. Adding a new node only
requires creating the node file - no manual registration needed.

See hexdag.stdlib.nodes._discovery for the auto-discovery mechanism.
"""

from .agent_node import ReActAgentNode
from .api_call_node import ApiCallNode
from .composite_node import CompositeNode
from .data_node import DataNode
from .expression_node import ExpressionNode
from .function_node import FunctionNode
from .llm_node import LLMNode
from .port_call_node import PortCallNode
from .tool_call_node import ToolCallNode

__all__ = [
    "ApiCallNode",
    "CompositeNode",
    "DataNode",
    "ExpressionNode",
    "FunctionNode",
    "LLMNode",
    "PortCallNode",
    "ReActAgentNode",
    "ToolCallNode",
]

# Bootstrap: Register auto-discovered node aliases with core resolver
# This maintains hexagonal architecture - builtin calls into core, not vice versa
from hexdag.kernel.resolver import register_builtin_aliases
from hexdag.stdlib.nodes._discovery import discover_node_factories

register_builtin_aliases(discover_node_factories())
