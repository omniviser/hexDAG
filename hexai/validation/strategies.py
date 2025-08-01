"""Validation strategies for the unified validation framework."""

from enum import Enum


class ValidationStrategy(Enum):
    """Validation strategies for different validation modes.

    - STRICT: Fail fast on any type mismatch or validation error
    - COERCE: Attempt type conversion and coercion when possible
    - PASSTHROUGH: Minimal validation, allow data to pass through unchanged
    """

    STRICT = "strict"
    COERCE = "coerce"
    PASSTHROUGH = "passthrough"

    def __str__(self) -> str:
        """Return the string representation of the validation strategy."""
        return self.value
