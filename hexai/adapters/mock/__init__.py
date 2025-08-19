"""Enhanced mock implementations for testing purposes.

Mock adapters for testing (lazy, conditional imports).
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_MAP: dict[str, tuple[str, str]] = {
    "MockLLM": ("hexai.adapters.mock.mock_llm", "MockLLM"),
    "MockDatabaseAdapter": ("hexai.adapters.mock.mock_database", "MockDatabaseAdapter"),
    "MockEmbeddingSelectorPort": (
        "hexai.adapters.mock.mock_embedding_selector",
        "MockEmbeddingSelectorPort",
    ),
}

__all__ = list(_LAZY_MAP.keys())


def __getattr__(name: str) -> Any:
    if name in _LAZY_MAP:
        module_name, attr = _LAZY_MAP[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__} has no attribute {name}")
