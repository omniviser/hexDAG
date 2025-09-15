"""Hex-DAG Agent Framework.

A modular, deterministic, and extensible architecture for orchestrating LLM-powered agents alongside
traditional code with YAML pipeline configuration.
"""

# Core framework exports
from hexai.core.application.nodes import ConditionalNode, FunctionNode, LoopNode
from hexai.core.application.orchestrator import Orchestrator
from hexai.core.domain import DirectedGraph, NodeSpec

# Initialize the component registry early so it's available for all imports
# This ensures core components are loaded and plugins are discovered
from hexai.core.registry import registry
from hexai.core.registry.models import ComponentType  # For internal framework use

# Load core components - using attribute assignment for dynamic loading flag
registry._core_loading = True  # type: ignore  # Allow core namespace registration
import hexai.core.application.nodes  # noqa: F401, E402 - triggers decorator registration

registry._core_loading = False  # type: ignore  # Block core namespace registration


__all__ = [
    # Registry System
    "registry",
    "ComponentType",
    # Core Framework - DAG Building and Execution
    "Orchestrator",
    "DirectedGraph",
    "NodeSpec",
    "FunctionNode",
    "LoopNode",
    "ConditionalNode",
]
