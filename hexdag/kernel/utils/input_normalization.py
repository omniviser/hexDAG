"""Shared input normalization for node wrappers.

Eliminates the repeated pattern across composite_node, function_node, etc.:
    if hasattr(data, "model_dump"):
        data = data.model_dump()
    elif isinstance(data, dict):
        data = dict(data)
    else:
        data = {"input": data}
"""

from typing import Any


def normalize_input(data: Any) -> dict[str, Any]:
    """Normalize input data to a plain dict.

    Handles Pydantic models (via ``model_dump``), dicts (shallow copy),
    and arbitrary values (wrapped as ``{"input": data}``).

    Examples
    --------
    >>> normalize_input({"key": "value"})
    {'key': 'value'}
    >>> normalize_input("hello")
    {'input': 'hello'}
    """
    if hasattr(data, "model_dump"):
        result: dict[str, Any] = data.model_dump()
        return result
    if isinstance(data, dict):
        return dict(data)
    return {"input": data}
