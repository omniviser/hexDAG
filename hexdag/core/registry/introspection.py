"""Optional introspection utilities for port/adapter validation.

⚠️  OPTIONAL MODULE - Not required for core registry functionality.

This module provides utilities for runtime introspection of ports and adapters.
It's primarily used in tests for validating port/adapter compatibility and may
be useful for CLI tooling or debugging.

Core registry operations (registration, lookup, bootstrap) do NOT depend on
this module - they use decorator metadata instead.

Use cases:
- Testing: Validate adapters implement required port methods
- CLI tools: Inspect port requirements and adapter capabilities
- Debugging: Extract tool signatures and port methods dynamically

If you're building a plugin or adapter, you probably don't need this module.
Use the @adapter and @port decorators instead, which handle metadata automatically.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


def extract_port_methods(port_class: type) -> tuple[list[str], list[str]]:
    """Extract required and optional methods from a Protocol class.

    Methods decorated with @abstractmethod are required.
    Methods with implementations are optional.

    Parameters
    ----------
    port_class : type
        The Protocol class to introspect

    Returns
    -------
    tuple[list[str], list[str]]
        (required_methods, optional_methods)

    Examples
    --------
    Example usage::

        from abc import abstractmethod
        @runtime_checkable
        class DatabasePort(Protocol):
        @abstractmethod
        def query(self, sql: str) -> list: ...
        @abstractmethod
        def execute(self, sql: str) -> None: ...
        def batch_execute(self, statements: list[str]) -> None:
        for stmt in statements:
        self.execute(stmt)
        required, optional = extract_port_methods(DatabasePort)
        assert required == ["query", "execute"]
        assert optional == ["batch_execute"]
    """
    required = []
    optional = []

    for name, method in inspect.getmembers(port_class, predicate=inspect.isfunction):
        # Skip private/special methods
        if name.startswith("_"):
            continue

        # Check if method is abstract
        # For Protocol classes, check if method has __isabstractmethod__ attribute
        is_abstract = hasattr(method, "__isabstractmethod__") and getattr(
            method, "__isabstractmethod__", False
        )
        if is_abstract:
            required.append(name)
        else:
            optional.append(name)

    return required, optional


def extract_tool_signature(func: Callable) -> dict[str, object]:
    """Extract tool information from function signature.

    Parameters
    ----------
    func : Callable
        The tool function to introspect

    Returns
    -------
    dict[str, Any]
        Tool information including:
        - is_async: bool
        - parameters: list of parameter info dicts
        - return_type: str representation of return type

    Examples
    --------
    Example usage::

        async def search(query: str, limit: int = 10) -> list[dict]:
        pass
        info = extract_tool_signature(search)
        assert info["is_async"] is True
        assert len(info["parameters"]) == 2
        assert info["parameters"][0]["name"] == "query"
        assert info["parameters"][0]["required"] is True
        assert info["parameters"][1]["default"] == 10
    """
    sig = inspect.signature(func)

    parameters = []
    for param_name, param in sig.parameters.items():
        param_info = {
            "name": param_name,
            "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any",
            "required": param.default == inspect.Parameter.empty,
            "default": param.default if param.default != inspect.Parameter.empty else None,
        }
        parameters.append(param_info)

    return {
        "is_async": asyncio.iscoroutinefunction(func),
        "parameters": parameters,
        "return_type": str(sig.return_annotation)
        if sig.return_annotation != inspect.Signature.empty
        else "Any",
    }


def validate_adapter_implementation(
    adapter_class: type,
    port_class: type,
) -> tuple[bool, list[str]]:
    """Validate that an adapter implements required port methods.

    Parameters
    ----------
    adapter_class : type
        The adapter class to validate
    port_class : type
        The port Protocol it should implement

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, missing_methods)

    Examples
    --------
    Example usage::

        class SQLiteAdapter:
        def query(self, sql: str) -> list:
        return []
        # If execute is required but missing:
        valid, missing = validate_adapter_implementation(SQLiteAdapter, DatabasePort)
        assert not valid
        assert "execute" in missing
    """
    required_methods, _ = extract_port_methods(port_class)

    missing = [method for method in required_methods if not hasattr(adapter_class, method)]

    return len(missing) == 0, missing


def infer_adapter_capabilities(
    adapter_class: type,
    port_class: type,
) -> list[str]:
    """Infer adapter capabilities based on optional methods implemented.

    Parameters
    ----------
    adapter_class : type
        The adapter class to check
    port_class : type
        The port Protocol it implements

    Returns
    -------
    list[str]
        List of capability names (e.g., ["supports_batch_execute"])

    Examples
    --------
    Example usage::

        class SQLiteAdapter:
        def query(self, sql: str) -> list:
        return []
        def execute(self, sql: str) -> None:
        pass
        def batch_execute(self, statements: list[str]) -> None:
        pass
        capabilities = infer_adapter_capabilities(SQLiteAdapter, DatabasePort)
        assert "supports_batch_execute" in capabilities
    """
    _, optional_methods = extract_port_methods(port_class)

    return [f"supports_{method}" for method in optional_methods if hasattr(adapter_class, method)]
