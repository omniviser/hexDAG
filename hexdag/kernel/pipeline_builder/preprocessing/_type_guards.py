"""Shared type guards for preprocessing plugins."""

from typing import Any, TypeGuard


def _is_dict_config(value: Any) -> TypeGuard[dict[str, Any]]:
    """Type guard to verify value is a dictionary."""
    return isinstance(value, dict)
