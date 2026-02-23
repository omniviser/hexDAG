"""Serialization helpers for distributed execution.

Provides utilities for checking whether values can safely cross process
boundaries (e.g., sent to a Celery worker via JSON).
"""

from typing import Any

# Primitive types that are JSON-safe and can cross process boundaries.
_JSON_SAFE_TYPES: frozenset[type] = frozenset({str, int, float, bool, type(None)})


def is_json_serializable(obj: Any) -> bool:
    """Recursively check that a value contains only JSON-safe types.

    Accepts: str, int, float, bool, None, list, tuple, and dict with str keys.
    Rejects: callables, Pydantic models, custom objects, bytes, etc.

    Parameters
    ----------
    obj : Any
        Value to check.

    Returns
    -------
    bool
        True if the entire value tree is JSON-serializable.

    Examples
    --------
    >>> is_json_serializable({"prompt": "hello", "retries": 3})
    True
    >>> is_json_serializable({"fn": lambda x: x})
    False
    """
    if type(obj) in _JSON_SAFE_TYPES:
        return True
    if isinstance(obj, (list, tuple)):
        return all(is_json_serializable(v) for v in obj)
    if isinstance(obj, dict):
        return all(isinstance(k, str) and is_json_serializable(v) for k, v in obj.items())
    return False
