"""Initalize the hexdag adapter package."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .unified_tool_router import UnifiedToolRouter  # noqa: F401

_LAZY_MAP: dict[str, tuple[str, str]] = {
    "UnifiedToolRouter": (
        "hexdag.adapters.unified_tool_router",
        "UnifiedToolRouter",
    ),
    # Backward compatibility alias
    "FunctionBasedToolRouter": (
        "hexdag.adapters.unified_tool_router",
        "UnifiedToolRouter",
    ),
}

__all__: tuple[str, ...] = tuple(_LAZY_MAP)


def __getattr__(name: str) -> Any:
    """Lazy import for adapters.

    Returns
    -------
    Any
        The imported adapter class or module.

    Raises
    ------
    AttributeError
        If the requested name is not found in the lazy import map.
    """
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
