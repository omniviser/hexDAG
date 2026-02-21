"""Initialize the hexdag adapter package."""

from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .unified_tool_router import UnifiedToolRouter  # noqa: F401

# Backward compatibility aliases: old name -> canonical name
_COMPAT_ALIASES: dict[str, str] = {
    "FunctionBasedToolRouter": "UnifiedToolRouter",
}

__all__: tuple[str, ...] = ("UnifiedToolRouter",)


def __getattr__(name: str) -> Any:
    """Lazy auto-discovery import for adapters.

    Scans top-level modules in this package to find the requested attribute.
    Results are cached in module globals for subsequent access.

    Returns
    -------
    Any
        The imported adapter class or module.

    Raises
    ------
    AttributeError
        If the requested name is not found in any adapter module.
    """
    # Resolve backward-compat aliases
    canonical = _COMPAT_ALIASES.get(name, name)

    # Scan top-level modules in this package
    package = importlib.import_module(__name__)
    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.name.startswith("_") or module_info.ispkg:
            continue
        module = importlib.import_module(f"{__name__}.{module_info.name}")
        if hasattr(module, canonical):
            value = getattr(module, canonical)

            # Register the port dynamically if necessary
            if hasattr(value, "register_port"):
                value.register_port()

            # Cache both canonical and alias names
            globals()[canonical] = value
            if name != canonical:
                globals()[name] = value
            return value

    raise AttributeError(f"module {__name__} has no attribute {name}")
