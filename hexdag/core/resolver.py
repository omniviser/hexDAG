"""Simple module path resolver for hexDAG components.

This replaces the registry system with Python's import system.
Components are resolved by their full module path.

Examples
--------
>>> from hexdag.core.resolver import resolve
>>> LLMNode = resolve("hexdag.builtin.nodes.LLMNode")
>>> adapter = resolve("hexdag.builtin.adapters.mock.MockLLM")
"""

from __future__ import annotations

import importlib
from typing import Any


class ResolveError(Exception):
    """Raised when a module path cannot be resolved."""

    def __init__(self, kind: str, reason: str):
        self.kind = kind
        self.reason = reason
        super().__init__(f"Cannot resolve '{kind}': {reason}")


# Runtime storage for dynamically created components (e.g., YAML-defined macros)
_runtime_components: dict[str, type[Any]] = {}

# User-registered aliases (separate from built-in short names)
_user_aliases: dict[str, str] = {}

# Backwards compatibility: map old short names to full module paths
_SHORT_NAME_ALIASES: dict[str, str] = {
    # Nodes (old short names)
    "llm_node": "hexdag.builtin.nodes.LLMNode",
    "function_node": "hexdag.builtin.nodes.FunctionNode",
    "agent_node": "hexdag.builtin.nodes.ReActAgentNode",
    "loop_node": "hexdag.builtin.nodes.LoopNode",
    "conditional_node": "hexdag.builtin.nodes.ConditionalNode",
    "tool_call_node": "hexdag.builtin.nodes.ToolCallNode",
    # Namespace:name format
    "core:llm_node": "hexdag.builtin.nodes.LLMNode",
    "core:function_node": "hexdag.builtin.nodes.FunctionNode",
    "core:agent_node": "hexdag.builtin.nodes.ReActAgentNode",
    "core:loop_node": "hexdag.builtin.nodes.LoopNode",
    "core:conditional_node": "hexdag.builtin.nodes.ConditionalNode",
}


def register_runtime(name: str, component: type[Any]) -> None:
    """Register a component created at runtime (e.g., YAML-defined macros).

    Parameters
    ----------
    name : str
        Component name (will be prefixed with 'runtime:')
    component : type
        The component class
    """
    _runtime_components[name] = component


def get_runtime(name: str) -> type[Any] | None:
    """Get a runtime-registered component.

    Parameters
    ----------
    name : str
        Component name

    Returns
    -------
    type | None
        The component class or None if not found
    """
    return _runtime_components.get(name)


def register_alias(alias: str, full_path: str) -> None:
    """Register a user-defined alias for a component path.

    This allows using short names in YAML instead of full module paths.
    User aliases take precedence over built-in short name aliases.

    Parameters
    ----------
    alias : str
        Short name to use in YAML (e.g., "my_processor")
    full_path : str
        Full module path (e.g., "myapp.nodes.ProcessorNode")

    Examples
    --------
    >>> register_alias("my_processor", "myapp.nodes.ProcessorNode")
    >>> resolve("my_processor")  # doctest: +SKIP
    <class 'myapp.nodes.ProcessorNode'>

    >>> # Use in YAML:
    >>> # spec:
    >>> #   aliases:
    >>> #     my_processor: myapp.nodes.ProcessorNode
    >>> #   nodes:
    >>> #     - kind: my_processor  # Uses alias!
    """
    if not alias:
        raise ValueError("Alias cannot be empty")
    if not full_path:
        raise ValueError("Full path cannot be empty")
    _user_aliases[alias] = full_path


def unregister_alias(alias: str) -> bool:
    """Remove a user-registered alias.

    Parameters
    ----------
    alias : str
        The alias to remove

    Returns
    -------
    bool
        True if alias was removed, False if it didn't exist
    """
    if alias in _user_aliases:
        del _user_aliases[alias]
        return True
    return False


def get_registered_aliases() -> dict[str, str]:
    """Get a copy of all user-registered aliases.

    Returns
    -------
    dict[str, str]
        Copy of the alias -> full_path mapping
    """
    return dict(_user_aliases)


def clear_aliases() -> None:
    """Clear all user-registered aliases.

    This is primarily useful for testing.
    """
    _user_aliases.clear()


def resolve(kind: str) -> type[Any]:
    """Resolve a kind string to a Python class.

    Parameters
    ----------
    kind : str
        Full module path to the class (e.g., "hexdag.builtin.nodes.LLMNode")
        or a runtime component name (e.g., "my_macro")
        or a legacy short name (e.g., "llm_node", "core:llm_node")

    Returns
    -------
    type
        The resolved class

    Raises
    ------
    ResolveError
        If the module or class cannot be found

    Examples
    --------
    >>> resolve("hexdag.builtin.nodes.LLMNode")  # doctest: +SKIP
    <class 'hexdag.builtin.nodes.llm_node.LLMNode'>

    >>> resolve("myapp.nodes.MyProcessor")  # doctest: +SKIP
    <class 'myapp.nodes.MyProcessor'>

    >>> # Legacy short name support
    >>> resolve("llm_node")  # doctest: +SKIP
    <class 'hexdag.builtin.nodes.llm_node.LLMNode'>
    """
    # First check runtime components (e.g., YAML-defined macros)
    if runtime_component := get_runtime(kind):
        return runtime_component

    # Check user-registered aliases (higher priority than built-in)
    if kind in _user_aliases:
        kind = _user_aliases[kind]

    # Check backwards compatibility aliases for legacy short names
    if kind in _SHORT_NAME_ALIASES:
        kind = _SHORT_NAME_ALIASES[kind]

    if "." not in kind:
        raise ResolveError(
            kind,
            "Must be a full module path (e.g., 'hexdag.builtin.nodes.LLMNode') "
            "or a registered runtime component",
        )

    try:
        module_path, class_name = kind.rsplit(".", 1)
    except ValueError as e:
        raise ResolveError(kind, "Invalid format - expected 'module.path.ClassName'") from e

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ResolveError(kind, f"Module '{module_path}' not found: {e}") from e
    except ImportError as e:
        raise ResolveError(kind, f"Failed to import '{module_path}': {e}") from e

    try:
        cls = getattr(module, class_name)
    except AttributeError as e:
        available = [name for name in dir(module) if not name.startswith("_")]
        raise ResolveError(
            kind,
            f"Class '{class_name}' not found in '{module_path}'. "
            f"Available: {', '.join(available[:10])}",
        ) from e

    if not isinstance(cls, type):
        raise ResolveError(kind, f"'{class_name}' is not a class (got {type(cls).__name__})")

    return cls


def resolve_function(path: str) -> Any:
    """Resolve a path to a function or callable.

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
