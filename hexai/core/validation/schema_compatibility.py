"""Schema compatibility checker using the converter registry.

This module provides utilities to check if data can be converted between different schemas,
leveraging the extensible converter system.
"""

from .converters import get_registry
from .strategies import ValidationStrategy
from .types import ValidationContext, ValidationResult


class SchemaCompatibilityChecker:
    """Checks schema compatibility using the converter registry."""

    def __init__(self, strategy: ValidationStrategy = ValidationStrategy.COERCE):
        """Initialize the compatibility checker.

        Args:
        ----
            strategy: Validation strategy to use for compatibility checks
        """
        self.strategy = strategy

    def can_convert_schema(self, source_type: type, target_type: type) -> bool:
        """Check if conversion is possible between two schema types.

        Args
        ----
            source_type: Source schema type
            target_type: Target schema type

        Returns
        -------
            True if conversion is possible, False otherwise
        """
        # Exact type match
        if source_type == target_type:
            return True

        # Check if converter registry can handle it
        registry = get_registry()
        return registry.can_convert(source_type, target_type)

    def validate_compatibility(
        self,
        source_type: type,
        target_type: type,
        context: ValidationContext | None = None,
    ) -> ValidationResult:
        """Validate schema compatibility with detailed result.

        Args
        ----
            source_type: Source schema type
            target_type: Target schema type
            context: Validation context for error reporting

        Returns
        -------
            ValidationResult indicating compatibility status
        """
        if context is None:
            context = ValidationContext(validation_stage="schema_compatibility")

        # Check compatibility
        if self.can_convert_schema(source_type, target_type):
            return ValidationResult.success(
                data={"compatible": True, "source": source_type, "target": target_type}
            )

        # Handle based on strategy
        error_msg = (
            f"Schema incompatibility: Cannot convert {source_type.__name__} "
            f"to {target_type.__name__}"
        )

        if context.node_name:
            error_msg += f" for node '{context.node_name}'"

        if self.strategy == ValidationStrategy.PASSTHROUGH:
            return ValidationResult.success(
                data={"compatible": False, "source": source_type, "target": target_type},
                warnings=[error_msg],
            )
        else:
            return ValidationResult.failure(
                data={"compatible": False, "source": source_type, "target": target_type},
                errors=[error_msg],
            )

    def suggest_conversions(
        self, source_type: type, available_targets: list[type]
    ) -> list[tuple[type, bool]]:
        """Suggest possible conversions from source to available target types.

        Args
        ----
            source_type: Source type to convert from
            available_targets: List of potential target types

        Returns
        -------
            List of (target_type, can_convert) tuples
        """
        suggestions = []
        for target_type in available_targets:
            can_convert = self.can_convert_schema(source_type, target_type)
            suggestions.append((target_type, can_convert))

        # Sort by compatibility (compatible first)
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions


# Convenience functions
def can_convert_schema(source_type: type, target_type: type) -> bool:
    """Check if schema conversion is possible.

    Args
    ----
        source_type: Source schema type
        target_type: Target schema type

    Returns
    -------
        True if conversion is possible, False otherwise
    """
    checker = SchemaCompatibilityChecker()
    return checker.can_convert_schema(source_type, target_type)
