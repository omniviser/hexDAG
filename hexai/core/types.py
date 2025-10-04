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

from typing import Annotated, Any, get_args, get_origin

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
# Framework Type Aliases (Python 3.12+ type statement)
# ============================================================================

# These use the modern `type` statement (PEP 695) instead of TypeAlias
# Note: Using Any here to avoid circular imports and external dependencies

# Logger type (loguru.Logger - using Any to avoid import)
type Logger = Any  # loguru.Logger

# Port types (to be replaced with proper Protocol definitions)
type PortInstance = Any  # Generic port implementation
type PortsDict = dict[str, PortInstance]  # Dictionary of port name -> port instance

# ============================================================================
# Type Inspection Utilities
# ============================================================================


def is_literal_type(type_hint: Any) -> bool:
    """Check if type hint is a Literal type.

    Examples
    --------
    >>> from typing import Literal
    >>> is_literal_type(Literal["a", "b"])
    True
    >>> is_literal_type(str)
    False
    """
    try:
        from typing import Literal

        origin = get_origin(type_hint)
        return origin is Literal
    except ImportError:
        return False


def is_union_type(type_hint: Any) -> bool:
    """Check if type hint is a Union type (including | syntax).

    Examples
    --------
    >>> from typing import Union
    >>> is_union_type(Union[str, int])
    True
    >>> is_union_type(str | int)
    True
    >>> is_union_type(str)
    False
    """
    origin = get_origin(type_hint)

    # Handle Union type
    try:
        from typing import Union

        if origin is Union:
            return True
    except ImportError:
        pass

    # Handle | syntax (Python 3.10+)
    try:
        from types import UnionType

        if isinstance(type_hint, UnionType):
            return True
    except ImportError:
        pass

    return False


def is_list_type(type_hint: Any) -> bool:
    """Check if type hint is a list type.

    Examples
    --------
    >>> is_list_type(list[str])
    True
    >>> is_list_type(list)
    True
    >>> is_list_type(str)
    False
    """
    origin = get_origin(type_hint)
    return origin is list or type_hint is list


def is_dict_type(type_hint: Any) -> bool:
    """Check if type hint is a dict type.

    Examples
    --------
    >>> is_dict_type(dict[str, int])
    True
    >>> is_dict_type(dict)
    True
    >>> is_dict_type(str)
    False
    """
    origin = get_origin(type_hint)
    return origin is dict or type_hint is dict


def is_annotated_type(type_hint: Any) -> bool:
    """Check if type hint is an Annotated type.

    Examples
    --------
    >>> from typing import Annotated
    >>> from pydantic import Field
    >>> is_annotated_type(Annotated[int, Field(ge=0)])
    True
    >>> is_annotated_type(int)
    False
    """
    try:
        from typing import Annotated

        origin = get_origin(type_hint)
        return origin is Annotated
    except ImportError:
        return False


def get_annotated_metadata(type_hint: Any) -> tuple[Any, tuple[Any, ...]]:
    """Extract base type and metadata from Annotated type.

    Examples
    --------
    >>> from typing import Annotated
    >>> from pydantic import Field
    >>> base, metadata = get_annotated_metadata(Annotated[int, Field(ge=0)])
    >>> base
    <class 'int'>
    """
    args = get_args(type_hint)
    if not args:
        return type_hint, ()

    base_type = args[0]
    metadata = args[1:] if len(args) > 1 else ()

    return base_type, metadata


# ============================================================================
# Secret Wrapper
# ============================================================================


class Secret:
    """Minimal secret wrapper to avoid accidental str() in logs.

    This class wraps sensitive string values to prevent them from being
    accidentally logged or printed. It's used by the SecretPort interface
    to return secret values in a safe manner.

    The Secret class uses name mangling (double underscore) to make it
    harder to accidentally access the raw value.

    Examples
    --------
    >>> secret = Secret("my-api-key")
    >>> print(secret)  # Safe - won't print the value
    <SECRET>
    >>> str(secret)  # Safe - won't convert to string
    '<SECRET>'
    >>> secret.get()  # Explicit access required
    'my-api-key'

    Usage in adapters::

        # SecretPort implementation
        async def aget_secret(self, key: str) -> Secret:
            value = os.getenv(key)
            return Secret(value)

        # Using the secret
        secret = await adapter.aget_secret("API_KEY")
        print(secret)  # <SECRET> (safe)
        api_key = secret.get()  # "sk-..." (explicit)
    """

    def __init__(self, value: str) -> None:
        """Initialize secret with a value.

        Parameters
        ----------
        value : str
            The secret value to wrap
        """
        self.__value = value  # Double underscore for name mangling

    def get(self) -> str:
        """Return the wrapped secret value securely.

        This is the only way to access the actual secret value.
        Requiring explicit .get() makes it clear when secrets are being accessed.

        Returns
        -------
        str
            The secret value
        """
        return self.__value

    def __repr__(self) -> str:
        """Return a safe string representation for debugging.

        Returns
        -------
        str
            A safe string representation "<SECRET>"
        """
        return "<SECRET>"

    def __str__(self) -> str:
        """Return a safe string representation for display.

        Returns
        -------
        str
            A safe string representation "<SECRET>"
        """
        return "<SECRET>"


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
    # Secret wrapper
    "Secret",
    # Framework types
    "Logger",
    "PortInstance",
    "PortsDict",
    # Type inspection utilities
    "is_literal_type",
    "is_union_type",
    "is_list_type",
    "is_dict_type",
    "is_annotated_type",
    "get_annotated_metadata",
]
