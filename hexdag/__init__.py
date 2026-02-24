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

# Templating system
# Primary API — the recommended entry point for running YAML pipelines
# Port interfaces (needed for writing custom adapters)
from hexdag.kernel import (
    LLM,
    APICall,
    Database,
    FewShotPromptTemplate,
    PipelineRunner,
    PromptTemplate,
    ToolRouter,
)
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
    from hexdag.drivers.http_client import HttpClientDriver
    from hexdag.kernel import DirectedGraph, NodeSpec, Orchestrator, resolve, resolve_function
    from hexdag.stdlib.adapters.memory import InMemoryMemory
    from hexdag.stdlib.adapters.mock import MockDatabaseAdapter, MockHttpClient, MockLLM


# Deprecated top-level exports — use submodule imports or PipelineRunner instead.
_DEPRECATED_IMPORTS: dict[str, tuple[str, str]] = {
    "Orchestrator": (
        "hexdag.kernel",
        "Use PipelineRunner or import from 'hexdag.kernel'.",
    ),
    "YamlPipelineBuilder": (
        "hexdag.compiler.yaml_builder",
        "Use PipelineRunner or import from 'hexdag.compiler.yaml_builder'.",
    ),
    "DirectedGraph": (
        "hexdag.kernel",
        "Import from 'hexdag.kernel' instead.",
    ),
    "NodeSpec": (
        "hexdag.kernel",
        "Import from 'hexdag.kernel' instead.",
    ),
    "resolve": (
        "hexdag.kernel",
        "Import from 'hexdag.kernel' instead.",
    ),
    "resolve_function": (
        "hexdag.kernel",
        "Import from 'hexdag.kernel' instead.",
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
    if name == "MockHttpClient":
        from hexdag.stdlib.adapters.mock import MockHttpClient as _MockHttpClient

        return _MockHttpClient
    if name == "HttpClientDriver":
        from hexdag.drivers.http_client import HttpClientDriver as _HttpClientDriver

        return _HttpClientDriver
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
    # Drivers
    "HttpClientDriver",
    # Testing and Development Adapters
    "InMemoryMemory",
    "MockLLM",
    "MockDatabaseAdapter",
    "MockHttpClient",
    # Deprecated (still importable with warning)
    "Orchestrator",
    "DirectedGraph",
    "NodeSpec",
    "YamlPipelineBuilder",
    "resolve",
    "resolve_function",
]
