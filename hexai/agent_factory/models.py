"""Domain models for pipeline applications.

This module contains data models specific to pipeline applications, particularly for ontology
analytics and text-to-SQL use cases.

Optimized with Pydantic v2 best practices:
- Annotated types for Rust-level validation
- Optimized model configurations
- TypeAdapter caching for bulk operations
"""

from typing import Annotated

from pydantic import ConfigDict, Field

# --- Optimized type definitions with Annotated constraints ---
# These push validation into pydantic-core (Rust) for better performance

PositiveInt = Annotated[int, Field(gt=0)]
NonEmptyStr = Annotated[str, Field(min_length=1, max_length=255)]
Identifier = Annotated[int, Field(gt=0)]
TenantId = Annotated[int, Field(gt=0)]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
QueryId = Annotated[str, Field(pattern=r"^[a-zA-Z0-9_-]+$", min_length=1, max_length=100)]

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
