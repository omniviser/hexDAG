"""Domain models for pipeline applications.

This module contains data models specific to pipeline applications, particularly for ontology
analytics and text-to-SQL use cases.

Optimized with Pydantic v2 best practices:
- Annotated types for Rust-level validation
- Optimized model configurations
- TypeAdapter caching for bulk operations
"""

from pydantic import ConfigDict

# Import centralized type definitions
from hexai.core.types import (
    PositiveInt,
)

# Legacy aliases for backward compatibility
Identifier = PositiveInt  # Use PositiveInt directly in new code

# --- Optimized base configuration ---
OPTIMIZED_CONFIG = ConfigDict(
    frozen=True,  # Immutable models (hashable, thread-safe)
    validate_default=False,  # Skip validation of default values
    validate_assignment=False,  # No re-validation on assignment (frozen anyway)
    use_enum_values=True,  # Use enum values directly
    cache_strings="keys",  # Cache string operations on keys
    extra="forbid",  # Fail fast on extra fields
    str_strip_whitespace=True,  # Auto-strip whitespace from strings
)
