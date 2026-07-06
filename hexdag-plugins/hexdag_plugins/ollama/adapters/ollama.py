"""DEPRECATED: OllamaAdapter moved to ``hexdag.stdlib.adapters.ollama``.

Install ``hexdag[ollama]`` and update imports / YAML adapter paths to
``hexdag.stdlib.adapters.ollama.OllamaAdapter`` (alias ``llm:ollama``).
This module re-exports the stdlib class for backward compatibility.
"""

import warnings

warnings.warn(
    "hexdag_plugins.ollama.OllamaAdapter has moved to "
    "hexdag.stdlib.adapters.ollama.OllamaAdapter; install hexdag[ollama] "
    "and update imports/YAML adapter paths (alias: llm:ollama).",
    DeprecationWarning,
    stacklevel=2,
)

from hexdag.stdlib.adapters.ollama.ollama_adapter import OllamaAdapter  # noqa: E402,F401

__all__ = ["OllamaAdapter"]
