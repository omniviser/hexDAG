"""DEPRECATED: VertexAIAdapter moved to ``hexdag.stdlib.adapters.google``.

Install ``hexdag[google]`` and update imports / YAML adapter paths to
``hexdag.stdlib.adapters.google.VertexAIAdapter`` (alias ``llm:vertex``).
This module re-exports the stdlib class for backward compatibility.
"""

import warnings

warnings.warn(
    "hexdag_plugins.google.VertexAIAdapter has moved to "
    "hexdag.stdlib.adapters.google.VertexAIAdapter; install hexdag[google] "
    "and update imports/YAML adapter paths (alias: llm:vertex).",
    DeprecationWarning,
    stacklevel=2,
)

from hexdag.stdlib.adapters.google.vertex_adapter import (  # noqa: E402,F401
    VertexAIAdapter,
    _pydantic_schema_to_gemini,
    aclose_all_vertex_clients,
)

__all__ = ["VertexAIAdapter", "aclose_all_vertex_clients"]
