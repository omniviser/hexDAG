"""Initalize the hexdag adapter package."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .function_tool_router import FunctionBasedToolRouter  # noqa: F401
    from .in_memory_memory import InMemoryMemory  # noqa: F401

_LAZY_MAP: dict[str, tuple[str, str]] = {
    "InMemoryMemory": ("hexai.adapters.in_memory_memory", "InMemoryMemory"),
    "FunctionBasedToolRouter": (
        "hexai.adapters.function_tool_router",
        "FunctionBasedToolRouter",
    ),
}

__all__ = list(_LAZY_MAP.keys())


def __getattr__(name: str) -> Any:
    """Lazy import for adapters."""
    if name in _LAZY_MAP:
        module_name, attr = _LAZY_MAP[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr)

        # Register the port dynamically if necessary
        if hasattr(value, "register_port"):
            value.register_port()

        globals()[name] = value  # Cache the imported module in the globals for future access
        return value

    raise AttributeError(f"module {__name__} has no attribute {name}")
