"""Unified validation framework for HexAI.

This module provides a unified approach to validation across all node types
and execution contexts, with configurable strategies and extensible architecture.

Key Components:
- ValidationStrategy: Enum for validation modes (strict, coerce, passthrough)
- ValidationResult: Standardized validation result object
- ValidationContext: Runtime context for validation operations
- IValidator: Abstract interface for all validators
- BaseValidator: Base implementation with common validation logic

Example Usage:
    from hexai.validation import ValidationStrategy, BaseValidator, ValidationContext

    # Create a validator with coerce strategy
    validator = BaseValidator(ValidationStrategy.COERCE)

    # Create validation context
    context = ValidationContext(node_name="text2sql", pipeline_name="main")

    # Validate input data
    result = validator.validate_input(input_data, str, context)

    if result.is_valid:
        print(f"Validation successful: {result.data}")
    else:
        print(f"Validation failed: {result.errors}")
"""

import importlib
import inspect

from .basic_converters import BooleanConverter, StringToNumericConverter
from .converters import ConversionRegistry, TypeConverter, get_registry, register_converter
from .core import BaseValidator, IValidator
from .errors import ConversionError, FallbackError, SchemaError, ValidationError
from .factory import ValidatorFactory, coerce_validator, passthrough_validator, strict_validator
from .pydantic_converters import (
    DictToPydanticConverter,
    JsonStringToPydanticConverter,
    NestedPydanticConverter,
    PydanticToDictConverter,
    PydanticToJsonConverter,
)
from .schema_compatibility import SchemaCompatibilityChecker, can_convert_schema
from .strategies import ValidationStrategy
from .types import ValidationContext, ValidationResult

__all__ = [
    # Core interfaces and implementations
    "IValidator",
    "BaseValidator",
    # Validation types
    "ValidationStrategy",
    "ValidationResult",
    "ValidationContext",
    # Factory and convenience functions
    "ValidatorFactory",
    "strict_validator",
    "coerce_validator",
    "passthrough_validator",
    # Type conversion
    "TypeConverter",
    "ConversionRegistry",
    "get_registry",
    "register_converter",
    # Extension utilities
    "register_custom_converter",
    "can_convert_types",
    "discover_converters",
    # Schema compatibility
    "SchemaCompatibilityChecker",
    "can_convert_schema",
    # Basic converters
    "StringToNumericConverter",
    "BooleanConverter",
    # Pydantic converters
    "DictToPydanticConverter",
    "PydanticToDictConverter",
    "JsonStringToPydanticConverter",
    "PydanticToJsonConverter",
    "NestedPydanticConverter",
    # Exceptions
    "ValidationError",
    "ConversionError",
    "SchemaError",
    "FallbackError",
]


# Auto-register built-in converters
def _register_builtin_converters() -> None:
    """Register built-in converters with the global registry."""
    from .basic_converters import BooleanConverter, StringToNumericConverter
    from .converters import get_registry
    from .pydantic_converters import (
        DictToPydanticConverter,
        JsonStringToPydanticConverter,
        NestedPydanticConverter,
        PydanticToDictConverter,
        PydanticToJsonConverter,
    )

    registry = get_registry()

    # Register basic converters
    registry.register(StringToNumericConverter())
    registry.register(BooleanConverter())

    # Register Pydantic converters
    registry.register(DictToPydanticConverter())
    registry.register(PydanticToDictConverter())
    registry.register(JsonStringToPydanticConverter())
    registry.register(PydanticToJsonConverter())
    registry.register(NestedPydanticConverter())


# Lazy initialization - only register when first accessed
_converters_registered = False


def _ensure_converters_registered() -> None:
    """Ensure built-in converters are registered."""
    global _converters_registered
    if not _converters_registered:
        _register_builtin_converters()
        _converters_registered = True


# Auto-register on module import
_ensure_converters_registered()


# Extension utilities for custom converters
def register_custom_converter(converter: TypeConverter) -> None:
    """Register a custom converter with the validation framework.

    This is the main entry point for users to add their own converters.

    Args
    ----
        converter: Custom converter instance implementing TypeConverter interface

    Example
    -------
        from hexai.validation import register_custom_converter, TypeConverter

        class MyCustomConverter(TypeConverter):
            def can_convert(self, source_type: type, target_type: type) -> bool:
                return source_type == MyType and target_type == str

            def convert(self, value: Any, target_type: type) -> Any:
                return str(value)

        register_custom_converter(MyCustomConverter())
    """
    _ensure_converters_registered()
    get_registry().register(converter)


def can_convert_types(source_type: type, target_type: type) -> bool:
    """Check if conversion is possible between two types.

    Args
    ----
        source_type: Source type to convert from
        target_type: Target type to convert to

    Returns
    -------
    -------
        True if conversion is possible, False otherwise

    Example:
    --------
        if can_convert_types(dict, UserModel):
            # Conversion is possible
            pass
    """
    _ensure_converters_registered()
    return get_registry().can_convert(source_type, target_type)


def discover_converters(module_name: str | None = None) -> list[TypeConverter]:
    """Discover and optionally auto-register converters from a module.

    Args
    ----
        module_name: Module to search for converters (optional)

    Returns
    -------
        List of discovered converter instances

    Example
    -------
        # Auto-discover from current package
        converters = discover_converters("my_app.converters")

        # Register all discovered converters
        for converter in converters:
            register_custom_converter(converter)
    """
    converters = []

    if module_name:
        try:
            module = importlib.import_module(module_name)
            for _name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, TypeConverter) and obj != TypeConverter:
                    # Try to instantiate the converter
                    try:
                        instance = obj()
                        converters.append(instance)
                    except (TypeError, ValueError, ImportError):
                        # Skip converters that can't be instantiated due to missing dependencies
                        # or invalid constructor arguments
                        continue
        except ImportError:
            # Module not found, return empty list
            pass

    return converters
