"""Core exception hierarchy for hexDAG framework.

This module provides a centralized exception hierarchy to reduce scattered
ValueError/TypeError raises throughout the codebase. All hexDAG exceptions
inherit from HexDAGError for easy exception handling.
"""

from __future__ import annotations

# ============================================================================
# Base Exception
# ============================================================================


class HexDAGError(Exception):
    """Base exception for all hexDAG framework errors.

    This is the root exception that all hexDAG-specific exceptions inherit from.
    Catch this to handle all hexDAG errors.
    """

    pass


# ============================================================================
# Configuration & Validation Errors
# ============================================================================


class ConfigurationError(HexDAGError):
    """Raised when configuration is invalid or missing.

    Examples
    --------
    Example usage::

        raise ConfigurationError("pipeline", "YAML file not found")
    """

    def __init__(self, component: str, reason: str) -> None:
        """Initialize configuration error.

        Args
        ----
            component: Name of the component with invalid configuration
            reason: Explanation of what's wrong
        """
        super().__init__(f"Configuration error in '{component}': {reason}")
        self.component = component
        self.reason = reason


class ValidationError(HexDAGError):
    """Raised when data validation fails.

    This replaces scattered ValueError raises for validation failures.

    Examples
    --------
    Example usage::

        raise ValidationError("max_iterations", "must be positive", value=-1)
    """

    def __init__(self, field: str, constraint: str, value: object = None) -> None:
        """Initialize validation error.

        Args
        ----
            field: Name of the field that failed validation
            constraint: Description of the validation constraint
            value: The invalid value (optional)
        """
        if value is not None:
            msg = f"Validation failed for '{field}': {constraint} (got {value!r})"
        else:
            msg = f"Validation failed for '{field}': {constraint}"
        super().__init__(msg)
        self.field = field
        self.constraint = constraint
        self.value = value


class ParseError(HexDAGError):
    """Raised when LLM output parsing fails.

    Used by ParserNode when JSON/YAML/structured parsing fails.
    Contains helpful retry hints for fixing the prompt.

    Examples
    --------
    Example usage::

        raise ParseError("Failed to parse JSON from LLM output. Retry hints: ...")
    """

    pass


# ============================================================================
# Resource & Dependency Errors
# ============================================================================


class ResourceNotFoundError(HexDAGError):
    """Raised when a required resource cannot be found.

    This replaces ValueError for missing files, pipelines, etc.

    Examples
    --------
    Example usage::

        raise ResourceNotFoundError("pipeline", "my_workflow", ["workflow1", "workflow2"])
    """

    def __init__(
        self, resource_type: str, resource_id: str, available: list[str] | None = None
    ) -> None:
        """Initialize resource not found error.

        Args
        ----
            resource_type: Type of resource (e.g., "pipeline", "file", "adapter")
            resource_id: Identifier of the missing resource
            available: List of available resources (optional)
        """
        msg = f"{resource_type.title()} '{resource_id}' not found"
        if available:
            msg += f". Available: {', '.join(available[:5])}"
            if len(available) > 5:
                msg += f" ... and {len(available) - 5} more"
        super().__init__(msg)
        self.resource_type = resource_type
        self.resource_id = resource_id
        self.available = available


class DependencyError(HexDAGError):
    """Raised when a required dependency is missing or invalid.

    Examples
    --------
    Example usage::

        raise DependencyError("llm", "LLM port is required for agent nodes")
    """

    def __init__(self, dependency: str, reason: str) -> None:
        """Initialize dependency error.

        Args
        ----
            dependency: Name of the missing/invalid dependency
            reason: Why it's needed or what's wrong
        """
        super().__init__(f"Dependency error for '{dependency}': {reason}")
        self.dependency = dependency
        self.reason = reason


# ============================================================================
# Type Errors
# ============================================================================


class TypeMismatchError(HexDAGError):
    """Raised when a value has an unexpected type.

    This replaces TypeError for type checking failures.

    Examples
    --------
    Example usage::

        raise TypeMismatchError("component", str, dict)
    """

    def __init__(
        self, field: str, expected: type | str, actual: type | str, value: object = None
    ) -> None:
        """Initialize type mismatch error.

        Args
        ----
            field: Name of the field with wrong type
            expected: Expected type or description
            actual: Actual type or description
            value: The value with wrong type (optional)
        """
        exp_str = expected.__name__ if isinstance(expected, type) else str(expected)
        act_str = actual.__name__ if isinstance(actual, type) else str(actual)

        if value is not None:
            msg = f"Type mismatch for '{field}': expected {exp_str}, got {act_str} ({value!r})"
        else:
            msg = f"Type mismatch for '{field}': expected {exp_str}, got {act_str}"
        super().__init__(msg)
        self.field = field
        self.expected = expected
        self.actual = actual
        self.value = value


# ============================================================================
# Orchestration Errors
# ============================================================================


class OrchestratorError(HexDAGError):
    """Raised when orchestrator execution encounters an error.

    This includes node execution failures, wave execution problems,
    and other orchestration-level issues.

    Examples
    --------
    Example usage::

        raise OrchestratorError("Node 'fetch_data' failed: timeout")
    """

    pass


__all__ = [
    "HexDAGError",
    "ConfigurationError",
    "ValidationError",
    "ResourceNotFoundError",
    "DependencyError",
    "TypeMismatchError",
    "OrchestratorError",
]
