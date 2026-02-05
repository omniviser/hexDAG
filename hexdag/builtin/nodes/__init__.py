"""Node factories for the hexdag framework.

All BaseNodeFactory subclasses in this package are auto-discovered
and registered for YAML pipeline validation. Adding a new node only
requires creating the node file - no manual registration needed.

See hexdag.builtin.nodes._discovery for the auto-discovery mechanism.
"""

from .agent_node import ReActAgentNode
from .data_node import DataNode
from .expression_node import ExpressionNode
from .function_node import FunctionNode
from .llm_node import LLMNode
from .loop_node import ConditionalNode, LoopNode
from .port_call_node import PortCallNode
from .tool_call_node import ToolCallNode

__all__ = [
    "ConditionalNode",
    "DataNode",
    "ExpressionNode",
    "FunctionNode",
    "LLMNode",
    "LoopNode",
    "PortCallNode",
    "ReActAgentNode",
    "ToolCallNode",
]

# Bootstrap: Register auto-discovered node aliases with core resolver
# This maintains hexagonal architecture - builtin calls into core, not vice versa
from hexdag.builtin.nodes._discovery import discover_node_factories
from hexdag.core.resolver import register_builtin_aliases

register_builtin_aliases(discover_node_factories())
