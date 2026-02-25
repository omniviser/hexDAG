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
from .tool_call_node import ToolCallNode

__all__ = [
    "ApiCallNode",
    "CompositeNode",
    "DataNode",
    "ExpressionNode",
    "FunctionNode",
    "LLMNode",
    "ReActAgentNode",
    "ToolCallNode",
]

# Bootstrap: Register __init_subclass__-discovered node aliases with core resolver.
# Importing the node classes above triggers __init_subclass__ which populates _registry.
from hexdag.kernel.resolver import register_builtin_aliases
from hexdag.stdlib.nodes.base_node_factory import BaseNodeFactory

register_builtin_aliases(BaseNodeFactory._registry)
