"""Core types for the unified validation framework."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationContext:
    """Runtime context information for validation operations.

    Provides context about where and how validation is being performed to enable context-aware
    validation and error reporting.
    """

    node_name: str | None = None
    pipeline_name: str | None = None
    validation_stage: str = "unknown"  # input, output, intermediate
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_stage(self, stage: str) -> "ValidationContext":
        """Create a new context with updated validation stage."""
        return ValidationContext(
            node_name=self.node_name,
            pipeline_name=self.pipeline_name,
            validation_stage=stage,
            metadata=self.metadata.copy(),
        )


@dataclass
class ValidationResult:
    """Standardized result object for validation operations.

    Contains validation status, validated data, and any errors or warnings that occurred during the
    validation process.
    """

    is_valid: bool
    data: Any
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls, data: Any, warnings: list[str] | None = None) -> "ValidationResult":
        """Create a successful validation result."""
        return cls(
            is_valid=True,
            data=data,
            warnings=warnings or [],
        )

    @classmethod
    def failure(
        cls, data: Any, errors: list[str], warnings: list[str] | None = None
    ) -> "ValidationResult":
        """Create a failed validation result."""
        return cls(
            is_valid=False,
            data=data,
            errors=errors,
            warnings=warnings or [],
        )

    def add_error(self, error: str) -> None:
        """Add an error to the validation result."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add a warning to the validation result."""
        self.warnings.append(warning)

    def __bool__(self) -> bool:
        """Return True if validation was successful."""
        return self.is_valid
