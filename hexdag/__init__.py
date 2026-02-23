"""hexDAG — Operating system for AI agents.

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
# Templating system
from hexdag.kernel.orchestration.prompt import FewShotPromptTemplate, PromptTemplate

# Primary API — the recommended entry point for running YAML pipelines
from hexdag.kernel.pipeline_runner import PipelineRunner

# Port interfaces (needed for writing custom adapters)
from hexdag.kernel.ports import LLM, APICall, Database, ToolRouter
from hexdag.stdlib.nodes import (
    FunctionNode,
    LLMNode,
    ReActAgentNode,
)

# Backward-compat alias (deprecated: use Database)
DatabasePort = Database

# Define placeholders for lazy-loaded adapters to satisfy __all__ checking
# These will be replaced by __getattr__ when accessed
if TYPE_CHECKING:
    from hexdag.compiler.yaml_builder import YamlPipelineBuilder
    from hexdag.kernel.domain import DirectedGraph, NodeSpec
    from hexdag.kernel.orchestration.orchestrator import Orchestrator
    from hexdag.kernel.resolver import resolve, resolve_function
    from hexdag.stdlib.adapters.memory import InMemoryMemory
    from hexdag.stdlib.adapters.mock import MockDatabaseAdapter, MockLLM


# Deprecated top-level exports — use submodule imports or PipelineRunner instead.
_DEPRECATED_IMPORTS: dict[str, tuple[str, str]] = {
    "Orchestrator": (
        "hexdag.kernel.orchestration.orchestrator",
        "Use PipelineRunner or import from 'hexdag.kernel.orchestration.orchestrator'.",
    ),
    "YamlPipelineBuilder": (
        "hexdag.compiler.yaml_builder",
        "Use PipelineRunner or import from 'hexdag.compiler.yaml_builder'.",
    ),
    "DirectedGraph": (
        "hexdag.kernel.domain",
        "Import from 'hexdag.kernel.domain' instead.",
    ),
    "NodeSpec": (
        "hexdag.kernel.domain",
        "Import from 'hexdag.kernel.domain' instead.",
    ),
    "resolve": (
        "hexdag.kernel.resolver",
        "Import from 'hexdag.kernel.resolver' instead.",
    ),
    "resolve_function": (
        "hexdag.kernel.resolver",
        "Import from 'hexdag.kernel.resolver' instead.",
    ),
}


# Lazy loading for adapters, deprecated exports, and optional modules
def __getattr__(name: str) -> Any:
    """Lazy import for adapters and optional components.

    Raises
    ------
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
        from hexdag.stdlib.adapters.mock import MockLLM as _MockLLM

        return _MockLLM
    if name == "MockDatabaseAdapter":
        from hexdag.stdlib.adapters.mock import MockDatabaseAdapter as _MockDatabaseAdapter

        return _MockDatabaseAdapter
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
    "Database",
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
