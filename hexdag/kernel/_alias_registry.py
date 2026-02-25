"""Shared builtin-alias storage and lightweight resolution helpers.

This module is intentionally low-dependency so that both
``hexdag.kernel.resolver`` (reader) and stdlib ``__init_subclass__``
hooks (writers) can import it without creating circular imports.
"""

from __future__ import annotations

import importlib
from typing import Any

from hexdag.kernel.exceptions import ResolveError

# Builtin aliases registered by hexdag.stdlib during bootstrap
_builtin_aliases: dict[str, str] = {}


def register_builtin_aliases(aliases: dict[str, str]) -> None:
    """Register builtin component aliases.

    Called by ``HexDAGAdapter.__init_subclass__``,
    ``BaseNodeFactory.__init_subclass__``, and stdlib discovery helpers
    to push aliases into the shared registry.

    Parameters
    ----------
    aliases : dict[str, str]
        Mapping of alias -> full module path
    """
    _builtin_aliases.update(aliases)


def resolve_function(path: str) -> Any:
    """Resolve a dotted path to a function or callable.

    Parameters
    ----------
    path : str
        Full module path to the function (e.g., "json.loads")

    Returns
    -------
    Callable
        The resolved function

    Raises
    ------
    ResolveError
        If the module or function cannot be found
    """
    if "." not in path:
        raise ResolveError(
            path,
            "Must be a full module path (e.g., 'json.loads')",
        )

    module_path, func_name = path.rsplit(".", 1)

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ResolveError(path, f"Module '{module_path}' not found: {e}") from e
    except ImportError as e:
        raise ResolveError(path, f"Failed to import '{module_path}': {e}") from e

    try:
        func = getattr(module, func_name)
    except AttributeError as e:
        available = [name for name in dir(module) if not name.startswith("_")]
        raise ResolveError(
            path,
            f"'{func_name}' not found in '{module_path}'. Available: {', '.join(available[:10])}",
        ) from e

    if not callable(func):
        raise ResolveError(path, f"'{func_name}' is not callable (got {type(func).__name__})")

    return func
