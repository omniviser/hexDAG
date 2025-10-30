"""Hex-DAG Agent Framework.

A modular, deterministic, and extensible architecture for orchestrating LLM-powered agents alongside
traditional code with YAML pipeline configuration.
"""

from typing import TYPE_CHECKING, Any

# Version is defined in pyproject.toml and read dynamically
try:
    from importlib.metadata import version

    __version__ = version("hexdag")
except Exception:
    __version__ = "0.0.0.dev0"  # Fallback for development installs

# Core framework exports
from hexdag.builtin.nodes import (
    ConditionalNode,
    FunctionNode,
    LLMNode,
    LoopNode,
    ReActAgentNode,
)
from hexdag.core.domain import DirectedGraph, NodeSpec
from hexdag.core.orchestration.orchestrator import Orchestrator
from hexdag.core.orchestration.prompt import FewShotPromptTemplate, PromptTemplate

# YAML Workflow Builder
from hexdag.core.pipeline_builder.yaml_builder import YamlPipelineBuilder

# Port interfaces
from hexdag.core.ports import LLM, APICall, DatabasePort, ToolRouter

# This ensures core components are loaded and plugins are discovered
from hexdag.core.registry import registry
from hexdag.core.registry.models import ComponentType  # For internal framework use

# Load core components - using attribute assignment for dynamic loading flag
registry._core_loading = True  # type: ignore  # Allow core namespace registration
import hexdag.builtin.nodes  # noqa: F401, E402 - triggers decorator registration

registry._core_loading = False  # type: ignore  # Block core namespace registration

# Define placeholders for lazy-loaded adapters to satisfy __all__ checking
# These will be replaced by __getattr__ when accessed
if TYPE_CHECKING:
    from hexdag.builtin.adapters.memory import InMemoryMemory
    from hexdag.builtin.adapters.mock import MockDatabaseAdapter, MockLLM


# Lazy loading for adapters and optional modules to avoid circular imports
def __getattr__(name: str) -> Any:
    """Lazy import for adapters and optional components.

    Raises
    ------
    ImportError
        If visualization module is not available
    AttributeError
        If the requested attribute does not exist
    """
    # Mock adapters
    if name == "MockLLM":
        from hexdag.builtin.adapters.mock import MockLLM as _MockLLM

        return _MockLLM
    if name == "MockDatabaseAdapter":
        from hexdag.builtin.adapters.mock import MockDatabaseAdapter as _MockDatabaseAdapter

        return _MockDatabaseAdapter
    if name == "MockToolRouter":
        from hexdag.builtin.adapters.mock import MockToolRouter as _MockToolRouter

        return _MockToolRouter

    # Visualization components (optional)
    if name == "DAGVisualizer":
        try:
            from hexdag.visualization import DAGVisualizer as _DAGVisualizer

            return _DAGVisualizer
        except ImportError as e:
            raise ImportError(
                "Visualization module not available. Install with:\n"
                "  pip install hexdag[viz]\n"
                "  or\n"
                "  uv pip install hexdag[viz]"
            ) from e

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Version
    "__version__",
    # Registry System
    "registry",
    "ComponentType",
    # Core Framework - DAG Building and Execution
    "Orchestrator",
    "DirectedGraph",
    "NodeSpec",
    "YamlPipelineBuilder",
    # Node Factories
    "FunctionNode",
    "LLMNode",
    "ReActAgentNode",
    "LoopNode",
    "ConditionalNode",
    # Templating System
    "PromptTemplate",
    "FewShotPromptTemplate",
    # Port Interfaces
    "LLM",
    "APICall",
    "ToolRouter",
    "DatabasePort",
    # Testing and Development Adapters
    "InMemoryMemory",
    "MockLLM",
    "MockDatabaseAdapter",
]
