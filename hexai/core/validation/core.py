"""Core validation interfaces and implementations for the unified validation framework."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from .converters import get_registry
from .strategies import ValidationStrategy
from .types import ValidationContext, ValidationResult


class IValidator(ABC):
    """Abstract base class for all validators in the unified validation framework.

    Defines the contract that all validators must implement to provide consistent validation
    behavior across different strategies and contexts.
    """

    @abstractmethod
    def validate_input(
        self,
        data: Any,
        expected_type: type[Any] | None = None,
        context: ValidationContext | None = None,
    ) -> ValidationResult:
        """Validate input data before node execution.

        Parameters
        ----------
        data : Any
            The input data to validate
        expected_type : Type[Any] | None
            The expected type for the input data
        context : ValidationContext | None
            Runtime context for validation

        Returns
        -------
        ValidationResult
            Result of the validation operation
        """
        pass

    @abstractmethod
    def validate_output(
        self,
        data: Any,
        expected_type: type[Any] | None = None,
        context: ValidationContext | None = None,
    ) -> ValidationResult:
        """Validate output data after node execution.

        Parameters
        ----------
        data : Any
            The output data to validate
        expected_type : Type[Any] | None
            The expected type for the output data
        context : ValidationContext | None
            Runtime context for validation

        Returns
        -------
        ValidationResult
            Result of the validation operation
        """
        pass


class BaseValidator(IValidator):
    """Base implementation of IValidator with common validation logic.

    Provides shared functionality for type checking and error handling that can be extended by
    specific validation strategies.
    """

    def __init__(self, strategy: ValidationStrategy):
        """Initialize the base validator.

        Parameters
        ----------
        strategy : ValidationStrategy
            The validation strategy to use
        """
        self.strategy = strategy

    def _is_pydantic_model(self, type_hint: type[Any] | None) -> bool:
        """Check if a type hint is a Pydantic model."""
        if type_hint is None:
            return False
        try:
            return issubclass(type_hint, BaseModel)
        except TypeError:
            return False

    def _is_basic_type(self, type_hint: type[Any] | None) -> bool:
        """Check if a type hint is a basic Python type."""
        if type_hint is None:
            return False
        return type_hint in (str, int, float, bool, list, dict)

    def _validate_pydantic_model(
        self, data: Any, model_type: type[BaseModel], context: ValidationContext | None = None
    ) -> ValidationResult:
        """Validate data against a Pydantic model using converter registry."""
        # Try direct validation first
        try:
            if self.strategy == ValidationStrategy.STRICT:
                validated_data = model_type.model_validate(data, strict=True)
            else:
                validated_data = model_type.model_validate(data, strict=False)

            return ValidationResult.success(validated_data)

        except Exception as direct_error:
            # If direct validation fails, try converters (both STRICT and COERCE modes)
            # STRICT mode allows Pydantic-to-Pydantic conversions if structurally compatible
            if self.strategy in (ValidationStrategy.STRICT, ValidationStrategy.COERCE):
                from .converters import get_registry

                registry = get_registry()
                source_type = type(data)

                if registry.can_convert(source_type, model_type):
                    try:
                        converted = registry.convert(data, model_type)
                        if self.strategy == ValidationStrategy.COERCE:
                            warning = f"Converted {source_type.__name__} to {model_type.__name__}"
                            return ValidationResult.success(converted, warnings=[warning])
                        else:
                            # In strict mode, don't show conversion warning for Pydantic models
                            return ValidationResult.success(converted)
                    except Exception as conversion_error:
                        error_msg = f"Pydantic conversion failed: {str(conversion_error)}"
                        if context and context.node_name:
                            error_msg += f" in node '{context.node_name}'"
                        return ValidationResult.failure(data, [error_msg])

            # Handle original validation error
            error_msg = f"Pydantic validation failed: {str(direct_error)}"
            if context and context.node_name:
                error_msg += f" in node '{context.node_name}'"

            if self.strategy == ValidationStrategy.PASSTHROUGH:
                return ValidationResult.success(data, warnings=[error_msg])
            else:
                return ValidationResult.failure(data, [error_msg])

    def _validate_basic_type(
        self, data: Any, expected_type: type[Any], context: ValidationContext | None = None
    ) -> ValidationResult:
        """Validate data against a basic Python type using converter registry."""
        if isinstance(data, expected_type):
            return ValidationResult.success(data)

        # Try type conversion for coerce strategy
        if self.strategy == ValidationStrategy.COERCE:
            registry = get_registry()
            source_type = type(data)

            # Try using converter registry first
            if registry.can_convert(source_type, expected_type):
                try:
                    converted = registry.convert(data, expected_type)
                    warning = f"Converted {source_type.__name__} to {expected_type.__name__}"
                    return ValidationResult.success(converted, warnings=[warning])
                except Exception as e:
                    error_msg = f"Type conversion failed: {str(e)}"
                    if context and context.node_name:
                        error_msg += f" in node '{context.node_name}'"
                    return ValidationResult.failure(data, [error_msg])

            # Fallback to basic conversions for backward compatibility
            try:
                if expected_type is str:
                    converted = str(data)
                elif expected_type is int:
                    converted = int(data)
                elif expected_type is float:
                    converted = float(data)
                elif expected_type is bool:
                    converted = bool(data)
                elif expected_type is list:
                    converted = (
                        list(data)
                        if hasattr(data, "__iter__") and not isinstance(data, str)
                        else [data]
                    )
                elif expected_type is dict:
                    if hasattr(data, "model_dump"):
                        converted = data.model_dump()
                    elif hasattr(data, "__dict__"):
                        converted = data.__dict__
                    else:
                        converted = {"value": data}
                else:
                    raise ValueError(f"Cannot convert to {expected_type.__name__}")

                warning = f"Converted {type(data).__name__} to {expected_type.__name__}"
                return ValidationResult.success(converted, warnings=[warning])

            except (ValueError, TypeError) as e:
                error_msg = f"Type conversion failed: {str(e)}"
                if context and context.node_name:
                    error_msg += f" in node '{context.node_name}'"
                return ValidationResult.failure(data, [error_msg])

        # Strict or passthrough strategy - format error message based on context
        if context and context.validation_stage == "input" and context.node_name:
            error_msg = (
                f"Node expects input of type {expected_type.__name__} "
                f"but received {type(data).__name__} in node '{context.node_name}'"
            )
        elif context and context.validation_stage == "output" and context.node_name:
            error_msg = (
                f"Node expects output of type {expected_type.__name__} "
                f"but received {type(data).__name__} in node '{context.node_name}'"
            )
        else:
            # Default format for tests and simple validation
            error_msg = f"Expected {expected_type.__name__}, got {type(data).__name__}"

        if self.strategy == ValidationStrategy.PASSTHROUGH:
            return ValidationResult.success(data, warnings=[error_msg])
        else:
            return ValidationResult.failure(data, [error_msg])

    def validate_input(
        self,
        data: Any,
        expected_type: type[Any] | None = None,
        context: ValidationContext | None = None,
    ) -> ValidationResult:
        """Validate input data using the configured strategy."""
        if context is None:
            context = ValidationContext(validation_stage="input")
        else:
            context = context.with_stage("input")

        # No validation needed
        if expected_type is None:
            return ValidationResult.success(data)

        # Pydantic model validation
        if self._is_pydantic_model(expected_type):
            return self._validate_pydantic_model(data, expected_type, context)

        # Basic type validation
        if self._is_basic_type(expected_type):
            return self._validate_basic_type(data, expected_type, context)

        # Generic isinstance check for other types
        if isinstance(data, expected_type):
            return ValidationResult.success(data)

        if context and context.validation_stage == "input" and context.node_name:
            error_msg = (
                f"Node expects input of type {expected_type.__name__} "
                f"but received {type(data).__name__} in node '{context.node_name}'"
            )
        elif context and context.validation_stage == "output" and context.node_name:
            error_msg = (
                f"Node expects output of type {expected_type.__name__} "
                f"but received {type(data).__name__} in node '{context.node_name}'"
            )
        else:
            # Default format for tests and simple validation
            error_msg = f"Expected {expected_type.__name__}, got {type(data).__name__}"

        # At this point, strategy can only be STRICT or COERCE (PASSTHROUGH handled earlier)
        return ValidationResult.failure(data, [error_msg])

    def validate_output(
        self,
        data: Any,
        expected_type: type[Any] | None = None,
        context: ValidationContext | None = None,
    ) -> ValidationResult:
        """Validate output data using the configured strategy."""
        if context is None:
            context = ValidationContext(validation_stage="output")
        else:
            context = context.with_stage("output")

        # Reuse input validation logic for now
        # Future: Add output-specific validation rules
        return self.validate_input(data, expected_type, context)
