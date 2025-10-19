"""Node factories for the hexai framework.

This module exports all node factory classes that are automatically
registered with the ComponentRegistry during bootstrap.
"""

from .agent_node import ReActAgentNode
from .function_node import FunctionNode
from .llm_node import LLMNode  # Deprecated - use PromptNode + RawLLMNode + ParserNode
from .loop_node import ConditionalNode, LoopNode
from .parser_node import ParserNode
from .prompt_node import PromptNode
from .raw_llm_node import RawLLMNode
from .tool_call_node import ToolCallNode

__all__ = [
    "FunctionNode",
    "LLMNode",  # Deprecated
    "RawLLMNode",  # New: Pure LLM API call
    "PromptNode",  # New: Prompt building
    "ParserNode",  # New: Output parsing
    "ReActAgentNode",
    "LoopNode",
    "ConditionalNode",
    "ToolCallNode",  # New: Tool execution wrapper
]
