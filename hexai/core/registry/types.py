"""Type definitions for the registry system."""

from __future__ import annotations

from enum import StrEnum


class ComponentType(StrEnum):
    """Enumeration of component types in the registry."""

    NODE = "node"
    ADAPTER = "adapter"
    TOOL = "tool"
    POLICY = "policy"
    MEMORY = "memory"
    OBSERVER = "observer"


class NodeSubtype(StrEnum):
    """Subtypes for NODE components - important for DAG boundaries."""

    FUNCTION = "function"  # Simple function node
    LLM = "llm"  # LLM-based node (has prompt templates)
    AGENT = "agent"  # Agent node (has tools, multi-step reasoning)
    LOOP = "loop"  # Loop control node
    CONDITIONAL = "conditional"  # Conditional branching node
    PASSTHROUGH = "passthrough"  # Simple passthrough node


class Namespace(StrEnum):
    """Standard namespaces for component organization."""

    CORE = "core"  # Protected core components
    USER = "user"  # Default for user-defined components
    TEST = "test"  # For testing purposes
    PLUGIN = "plugin"  # For plugin components
