"""
Adapter implementations for external services (lazy, conditional imports).

Import adapters explicitly or via lazy attributes, e.g.:
    from hexai.adapters import InMemoryMemory
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_MAP: dict[str, tuple[str, str]] = {
    "InMemoryMemory": ("hexai.adapters.in_memory_memory", "InMemoryMemory"),
}

__all__ = list(_LAZY_MAP.keys())


def __getattr__(name: str) -> Any:
    if name in _LAZY_MAP:
        module_name, attr = _LAZY_MAP[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr)
        globals()[name] = value  # cache
        return value
    raise AttributeError(f"module {__name__} has no attribute {name}")
