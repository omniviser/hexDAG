"""Hex-DAG Agent Framework.

A modular, deterministic, and extensible architecture for orchestrating LLM-powered agents alongside
traditional code with YAML pipeline configuration.
"""

import warnings
from typing import TYPE_CHECKING, Any

# Version is defined in pyproject.toml and read dynamically
try:
    from importlib.metadata import version

    __version__ = version("hexdag")
except Exception:
    __version__ = "0.0.0.dev0"  # Fallback for development installs

# Node factories (still useful for programmatic DAG building)
from hexdag.builtin.nodes import (
    FunctionNode,
    LLMNode,
    ReActAgentNode,
)

# Templating system
from hexdag.core.orchestration.prompt import FewShotPromptTemplate, PromptTemplate

# Primary API — the recommended entry point for running YAML pipelines
from hexdag.core.pipeline_runner import PipelineRunner

# Port interfaces (needed for writing custom adapters)
from hexdag.core.ports import LLM, APICall, DatabasePort, ToolRouter

# Define placeholders for lazy-loaded adapters to satisfy __all__ checking
# These will be replaced by __getattr__ when accessed
if TYPE_CHECKING:
    from hexdag.builtin.adapters.memory import InMemoryMemory
    from hexdag.builtin.adapters.mock import MockDatabaseAdapter, MockLLM
    from hexdag.core.domain import DirectedGraph, NodeSpec
    from hexdag.core.orchestration.orchestrator import Orchestrator
    from hexdag.core.pipeline_builder.yaml_builder import YamlPipelineBuilder
    from hexdag.core.resolver import resolve, resolve_function


# Deprecated top-level exports — use submodule imports or PipelineRunner instead.
_DEPRECATED_IMPORTS: dict[str, tuple[str, str]] = {
    "Orchestrator": (
        "hexdag.core.orchestration.orchestrator",
        "Use PipelineRunner or import from 'hexdag.core.orchestration.orchestrator'.",
    ),
    "YamlPipelineBuilder": (
        "hexdag.core.pipeline_builder.yaml_builder",
        "Use PipelineRunner or import from 'hexdag.core.pipeline_builder.yaml_builder'.",
    ),
    "DirectedGraph": (
        "hexdag.core.domain",
        "Import from 'hexdag.core.domain' instead.",
    ),
    "NodeSpec": (
        "hexdag.core.domain",
        "Import from 'hexdag.core.domain' instead.",
    ),
    "resolve": (
        "hexdag.core.resolver",
        "Import from 'hexdag.core.resolver' instead.",
    ),
    "resolve_function": (
        "hexdag.core.resolver",
        "Import from 'hexdag.core.resolver' instead.",
    ),
}


# Lazy loading for adapters, deprecated exports, and optional modules
def __getattr__(name: str) -> Any:
    """Lazy import for adapters and optional components.

    Raises
    ------
    ImportError
        If visualization module is not available
    AttributeError
        If the requested attribute does not exist
    """
    # Deprecated exports — still work but emit a warning
    if name in _DEPRECATED_IMPORTS:
        import importlib

        module_path, message = _DEPRECATED_IMPORTS[name]
        warnings.warn(
            f"Importing {name!r} from 'hexdag' is deprecated. {message}",
            DeprecationWarning,
            stacklevel=2,
        )
        mod = importlib.import_module(module_path)
        return getattr(mod, name)

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
    # Primary API
    "PipelineRunner",
    # Node Factories
    "FunctionNode",
    "LLMNode",
    "ReActAgentNode",
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
    # Deprecated (still importable with warning)
    "Orchestrator",
    "DirectedGraph",
    "NodeSpec",
    "YamlPipelineBuilder",
    "resolve",
    "resolve_function",
]
