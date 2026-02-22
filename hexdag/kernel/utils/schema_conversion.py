"""Shared utilities for converting YAML schema strings to Python types.

This module provides utilities used by both YAML pipeline builders and node factories
to convert YAML-friendly type representations to actual Python types.
"""

from functools import singledispatch
from typing import Any

# Canonical string-name → Python-type mapping (single source of truth).
# Used by normalize_schema, pipeline builder, and node factories.
VALID_TYPE_NAMES: dict[str, Any] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "Any": Any,
}


def is_valid_type_name(name: str) -> bool:
    """Check if a type name is valid, including nullable variants and sanitized types."""
    if name in VALID_TYPE_NAMES:
        return True

    from hexdag.kernel.validation.sanitized_types import get_type

    base = name.rstrip("?")
    if get_type(base) is not None:
        return True
    return name.endswith("?") and base in VALID_TYPE_NAMES and base != "Any"


@singledispatch
def normalize_schema(schema: Any) -> Any:
    """Normalize schema to use Python types (accepts both string names and type objects).

    Uses singledispatch to handle different input types elegantly.

    Args:
        schema: Schema in various formats (dict, type, Pydantic model, etc.)

    Returns:
        Normalized schema with actual Python type objects

    Examples:
        >>> normalize_schema({"name": "str"})  # YAML format
        {'name': <class 'str'>}

        >>> normalize_schema({"name": str})  # Already normalized
        {'name': <class 'str'>}

        >>> normalize_schema(str)  # Pass-through for types
        <class 'str'>
    """
    # Default: pass through for types, Pydantic models, etc.
    return schema


@normalize_schema.register(dict)
def _(schema: dict) -> dict[str, type]:
    """Convert dict schema with string type names to Python types."""
    converted: dict[str, Any] = {}

    for key, value in schema.items():
        if isinstance(value, str):
            # String type name - convert to actual type
            if not is_valid_type_name(value):
                from hexdag.kernel.validation.sanitized_types import get_available_types

                valid_names = ", ".join(sorted(VALID_TYPE_NAMES.keys()))
                sanitized_names = ", ".join(sorted(get_available_types()))
                hint = f" Sanitized types: {sanitized_names}" if sanitized_names else ""
                raise ValueError(
                    f"Invalid type '{value}' for field '{key}'. "
                    f"Supported types: {valid_names} (append ? for nullable).{hint}"
                )
            # Base types → Python type; nullable (?-suffixed) → pass through
            converted[key] = VALID_TYPE_NAMES.get(value, value)
        elif isinstance(value, dict):
            # Nested schema - recurse
            converted[key] = normalize_schema(value)
        elif isinstance(value, type):
            # Already a type - pass through
            converted[key] = value
        else:
            raise ValueError(
                f"Field '{key}' has invalid value: {value!r}. "
                f"Expected type name string, type object, or nested schema dict."
            )

    return converted


# Backward compatibility alias
convert_yaml_schema = normalize_schema
