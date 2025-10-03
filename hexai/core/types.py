"""Central repository for reusable Annotated types.

This module provides type aliases with validation constraints that are used
throughout the hexDAG framework. Using Annotated types reduces boilerplate,
improves type safety, and provides better IDE support.

Type Categories
---------------
- **Numeric Constraints**: PositiveInt, NonNegativeInt, PositiveFloat, etc.
- **LLM Parameters**: Temperature, TopP, Penalty, etc.
- **Time/Duration**: TimeoutSeconds, DelaySeconds
- **String Constraints**: NonEmptyStr, Identifier, QueryId
- **Domain Types**: TenantId, Confidence

Examples
--------
```python
from hexai.core.types import Temperature01, TimeoutSeconds

class Config(BaseModel):
    temperature: Temperature01 = 0.7
    timeout: TimeoutSeconds = 60.0
```
"""

from typing import Annotated

from pydantic import Field

# ============================================================================
# Numeric Constraints
# ============================================================================

PositiveInt = Annotated[int, Field(gt=0, description="Positive integer (> 0)")]
"""Integer greater than 0."""

NonNegativeInt = Annotated[int, Field(ge=0, description="Non-negative integer (>= 0)")]
"""Integer greater than or equal to 0."""

PositiveFloat = Annotated[float, Field(gt=0, description="Positive float (> 0)")]
"""Float greater than 0."""

NonNegativeFloat = Annotated[float, Field(ge=0.0, description="Non-negative float (>= 0)")]
"""Float greater than or equal to 0."""

# ============================================================================
# LLM Parameters
# ============================================================================

Temperature01 = Annotated[
    float, Field(ge=0.0, le=1.0, description="Temperature sampling parameter [0.0, 1.0]")
]
"""LLM temperature parameter, range [0.0, 1.0]. Used by Anthropic/Claude."""

Temperature02 = Annotated[
    float, Field(ge=0.0, le=2.0, description="Temperature sampling parameter [0.0, 2.0]")
]
"""LLM temperature parameter, range [0.0, 2.0]. Used by OpenAI."""

TopP = Annotated[float, Field(ge=0.0, le=1.0, description="Top-p (nucleus) sampling parameter")]
"""Top-p sampling parameter, range [0.0, 1.0]."""

FrequencyPenalty = Annotated[
    float, Field(ge=-2.0, le=2.0, description="Frequency penalty [-2.0, 2.0]")
]
"""Frequency penalty for token repetition, range [-2.0, 2.0]."""

PresencePenalty = Annotated[
    float, Field(ge=-2.0, le=2.0, description="Presence penalty [-2.0, 2.0]")
]
"""Presence penalty for token diversity, range [-2.0, 2.0]."""

TokenCount = Annotated[int, Field(gt=0, description="Number of tokens")]
"""Token count, must be positive."""

# ============================================================================
# Time and Duration
# ============================================================================

TimeoutSeconds = Annotated[float, Field(gt=0, description="Timeout duration in seconds")]
"""Timeout duration in seconds, must be positive."""

DelaySeconds = Annotated[float, Field(ge=0.0, description="Delay duration in seconds")]
"""Delay duration in seconds, non-negative."""

# ============================================================================
# Retry and Resilience
# ============================================================================

RetryCount = Annotated[int, Field(ge=0, description="Maximum retry attempts")]
"""Maximum number of retry attempts, non-negative."""

# ============================================================================
# String Constraints
# ============================================================================

NonEmptyStr = Annotated[str, Field(min_length=1, max_length=255, description="Non-empty string")]
"""Non-empty string with max length 255."""

Identifier = Annotated[
    str,
    Field(
        pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$",
        min_length=1,
        max_length=100,
        description="Valid identifier (Python-style)",
    ),
]
"""Valid Python-style identifier."""

QueryId = Annotated[
    str,
    Field(
        pattern=r"^[a-zA-Z0-9_-]+$",
        min_length=1,
        max_length=100,
        description="Query identifier",
    ),
]
"""Query identifier with alphanumeric, underscore, and hyphen characters."""

FilePath = Annotated[str, Field(min_length=1, description="File system path")]
"""File system path, non-empty."""

# ============================================================================
# Domain-Specific Types
# ============================================================================

TenantId = Annotated[int, Field(gt=0, description="Tenant identifier")]
"""Unique tenant identifier, positive integer."""

Confidence = Annotated[float, Field(ge=0.0, le=1.0, description="Confidence score [0.0, 1.0]")]
"""Confidence score, range [0.0, 1.0]."""

Percentage = Annotated[float, Field(ge=0.0, le=100.0, description="Percentage [0.0, 100.0]")]
"""Percentage value, range [0.0, 100.0]."""

# ============================================================================
# Backward Compatibility (deprecated, use specific types above)
# ============================================================================

# These were originally defined in agent_factory/models.py
# Kept here for backward compatibility but prefer the specific types above
__all__ = [
    # Numeric
    "PositiveInt",
    "NonNegativeInt",
    "PositiveFloat",
    "NonNegativeFloat",
    # LLM
    "Temperature01",
    "Temperature02",
    "TopP",
    "FrequencyPenalty",
    "PresencePenalty",
    "TokenCount",
    # Time
    "TimeoutSeconds",
    "DelaySeconds",
    # Retry
    "RetryCount",
    # String
    "NonEmptyStr",
    "Identifier",
    "QueryId",
    "FilePath",
    # Domain
    "TenantId",
    "Confidence",
    "Percentage",
]
