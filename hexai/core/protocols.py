"""Protocols for structural typing in hexDAG.

This module defines protocols (structural types) to reduce isinstance() usage
and enable duck typing with static type checking support.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# ============================================================================
# Component Protocols
# ============================================================================


@runtime_checkable
class ComponentWithExecute(Protocol):
    """Protocol for components that have an execute method.

    This allows duck typing for tool classes without isinstance checks.

    Examples
    --------
    >>> class MyTool:
    ...     def execute(self, **kwargs: Any) -> Any:
    ...         return "result"
    >>>
    >>> def use_tool(tool: ComponentWithExecute) -> Any:
    ...     return tool.execute(param=1)
    """

    def execute(self, **kwargs: Any) -> Any:
        """Execute the component with given parameters."""
        ...


# ============================================================================
# Port Protocols
# ============================================================================


@runtime_checkable
class ConfigurablePort(Protocol):
    """Protocol for ports that support configuration.

    Examples
    --------
    >>> class MyAdapter:
    ...     @classmethod
    ...     def get_config_class(cls) -> type:
    ...         return MyConfig
    """

    @classmethod
    def get_config_class(cls) -> type[Any]:
        """Return the configuration class for this port."""
        ...


@runtime_checkable
class HealthCheckable(Protocol):
    """Protocol for components that support health checks.

    Examples
    --------
    >>> class MyAdapter:
    ...     async def ahealth_check(self) -> dict[str, Any]:
    ...         return {"status": "healthy"}
    """

    async def ahealth_check(self) -> dict[str, Any]:
        """Perform async health check.

        Returns
        -------
            Health status dictionary with at least 'status' key
        """
        ...


# ============================================================================
# Data Conversion Protocols
# ============================================================================


@runtime_checkable
class DictConvertible(Protocol):
    """Protocol for objects that can be converted to dict.

    This includes Pydantic models and other dict-like objects.

    Examples
    --------
    >>> class MyModel:
    ...     def model_dump(self) -> dict[str, Any]:
    ...         return {"field": "value"}
    """

    def model_dump(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        ...


@runtime_checkable
class SchemaProvider(Protocol):
    """Protocol for classes that provide schema information.

    Examples
    --------
    >>> class MyModel:
    ...     @classmethod
    ...     def model_json_schema(cls) -> dict[str, Any]:
    ...         return {"type": "object", "properties": {}}
    """

    @classmethod
    def model_json_schema(cls) -> dict[str, Any]:
        """Return JSON schema for this model."""
        ...


# ============================================================================
# Helper Functions
# ============================================================================


def has_execute_method(obj: Any) -> bool:
    """Check if object has an execute method (duck typing).

    This is more Pythonic than isinstance checks.

    Args
    ----
        obj: Object to check

    Returns
    -------
        True if object has callable execute method

    Examples
    --------
    >>> class Tool:
    ...     def execute(self): pass
    >>> has_execute_method(Tool())
    True
    """
    return isinstance(obj, ComponentWithExecute)


def is_dict_convertible(obj: Any) -> bool:
    """Check if object can be converted to dict.

    Args
    ----
        obj: Object to check

    Returns
    -------
        True if object has model_dump method or is a dict

    Examples
    --------
    >>> from pydantic import BaseModel
    >>> class MyModel(BaseModel):
    ...     field: str
    >>> is_dict_convertible(MyModel(field="value"))
    True
    >>> is_dict_convertible({"key": "value"})
    True
    """
    return isinstance(obj, (dict, DictConvertible))


def is_schema_type(type_obj: Any) -> bool:
    """Check if type is a Pydantic model class (not instance).

    Args
    ----
        type_obj: Type to check

    Returns
    -------
        True if type is a Pydantic BaseModel subclass

    Examples
    --------
    >>> from pydantic import BaseModel
    >>> class MyModel(BaseModel):
    ...     field: str
    >>> is_schema_type(MyModel)  # Type check
    True
    >>> is_schema_type(MyModel(field="val"))  # Instance check
    False
    """
    try:
        return isinstance(type_obj, type) and issubclass(type_obj, SchemaProvider)
    except TypeError:
        return False


__all__ = [
    "ComponentWithExecute",
    "ConfigurablePort",
    "HealthCheckable",
    "DictConvertible",
    "SchemaProvider",
    "has_execute_method",
    "is_dict_convertible",
    "is_schema_type",
]
