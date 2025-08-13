"""Pydantic model converters for the unified validation framework.

This module provides converters for working with Pydantic models, including conversion from/to
dictionaries, JSON strings, and handling nested model conversion.
"""

import json
from typing import Any

from pydantic import BaseModel, ValidationError

from .converters import ConversionError, TypeConverter


class DictToPydanticConverter(TypeConverter):
    """Converter from dictionary to Pydantic model."""

    def can_convert(self, source_type: type, target_type: type) -> bool:
        """Check if we can convert dict to Pydantic model."""
        return source_type is dict and self._is_pydantic_model(target_type)

    def convert(self, value: Any, target_type: type) -> BaseModel:
        """Convert dictionary to Pydantic model."""
        if not isinstance(value, dict):
            raise ConversionError("Value must be a dictionary", type(value), target_type, value)

        if not self._is_pydantic_model(target_type):
            raise ConversionError(
                f"Target type {target_type} is not a Pydantic model", dict, target_type, value
            )

        try:
            return target_type.model_validate(value)  # type: ignore[attr-defined,no-any-return]
        except ValidationError as e:
            error_details = []
            for error in e.errors():
                loc = ".".join(str(x) for x in error["loc"])
                error_details.append(f"{loc}: {error['msg']}")

            raise ConversionError(
                f"Pydantic validation failed: {'; '.join(error_details)}", dict, target_type, value
            ) from e

    def _is_pydantic_model(self, type_: type) -> bool:
        """Check if type is a Pydantic model."""
        try:
            return issubclass(type_, BaseModel)
        except TypeError:
            return False


class PydanticToDictConverter(TypeConverter):
    """Converter from Pydantic model to dictionary."""

    def can_convert(self, source_type: type, target_type: type) -> bool:
        """Check if we can convert Pydantic model to dict."""
        return target_type is dict and self._is_pydantic_model(source_type)

    def convert(self, value: Any, target_type: type) -> dict[str, Any]:
        """Convert Pydantic model to dictionary."""
        if not isinstance(value, BaseModel):
            raise ConversionError(
                "Value must be a Pydantic model instance", type(value), target_type, value
            )

        if target_type is not dict:
            raise ConversionError("Target type must be dict", type(value), target_type, value)

        try:
            return value.model_dump()
        except Exception as e:
            raise ConversionError(
                f"Failed to convert Pydantic model to dict: {str(e)}", type(value), dict, value
            ) from e

    def _is_pydantic_model(self, type_: type) -> bool:
        """Check if type is a Pydantic model."""
        try:
            return issubclass(type_, BaseModel)
        except TypeError:
            return False


class JsonStringToPydanticConverter(TypeConverter):
    """Converter from JSON string to Pydantic model."""

    def can_convert(self, source_type: type, target_type: type) -> bool:
        """Check if we can convert JSON string to Pydantic model."""
        return source_type is str and self._is_pydantic_model(target_type)

    def convert(self, value: Any, target_type: type) -> BaseModel:
        """Convert JSON string to Pydantic model."""
        if not isinstance(value, str):
            raise ConversionError("Value must be a string", type(value), target_type, value)

        if not self._is_pydantic_model(target_type):
            raise ConversionError(
                f"Target type {target_type} is not a Pydantic model", str, target_type, value
            )

        # First parse JSON string to dict
        try:
            data = json.loads(value)
        except json.JSONDecodeError as e:
            raise ConversionError(f"Invalid JSON string: {str(e)}", str, target_type, value) from e

        if not isinstance(data, dict):
            raise ConversionError(
                f"JSON must represent an object (dict), got {type(data).__name__}",
                str,
                target_type,
                value,
            )

        # Then convert dict to Pydantic model
        try:
            return target_type.model_validate(data)  # type: ignore[attr-defined,no-any-return]
        except ValidationError as e:
            error_details = []
            for error in e.errors():
                loc = ".".join(str(x) for x in error["loc"])
                error_details.append(f"{loc}: {error['msg']}")

            raise ConversionError(
                f"Pydantic validation failed: {'; '.join(error_details)}", str, target_type, value
            ) from e

    def _is_pydantic_model(self, type_: type) -> bool:
        """Check if type is a Pydantic model."""
        try:
            return issubclass(type_, BaseModel)
        except TypeError:
            return False


class PydanticToJsonConverter(TypeConverter):
    """Converter from Pydantic model to JSON string."""

    def can_convert(self, source_type: type, target_type: type) -> bool:
        """Check if we can convert Pydantic model to JSON string."""
        return target_type is str and self._is_pydantic_model(source_type)

    def convert(self, value: Any, target_type: type) -> str:
        """Convert Pydantic model to JSON string."""
        if not isinstance(value, BaseModel):
            raise ConversionError(
                "Value must be a Pydantic model instance", type(value), target_type, value
            )

        if target_type is not str:
            raise ConversionError("Target type must be str", type(value), target_type, value)

        try:
            return value.model_dump_json()
        except Exception as e:
            raise ConversionError(
                f"Failed to convert Pydantic model to JSON: {str(e)}", type(value), str, value
            ) from e

    def _is_pydantic_model(self, type_: type) -> bool:
        """Check if type is a Pydantic model."""
        try:
            return issubclass(type_, BaseModel)
        except TypeError:
            return False


class NestedPydanticConverter(TypeConverter):
    """Converter for nested Pydantic models."""

    def can_convert(self, source_type: type, target_type: type) -> bool:
        """Check if we can convert between Pydantic models."""
        if not (
            self._is_pydantic_model(source_type)
            and self._is_pydantic_model(target_type)
            and source_type != target_type
        ):
            return False

        # Test actual conversion compatibility by examining model fields
        try:
            source_fields = source_type.model_fields  # type: ignore[attr-defined]
            target_fields = target_type.model_fields  # type: ignore[attr-defined]

            # Check if all required target fields can be satisfied by source fields
            for field_name, field_info in target_fields.items():
                # If field is required and not in source, conversion will fail
                if field_info.is_required() and field_name not in source_fields:
                    return False

            return True
        except Exception:
            # If we can't analyze the fields, be conservative
            return False

    def convert(self, value: Any, target_type: type) -> BaseModel:
        """Convert between different Pydantic models via dict intermediate."""
        if not isinstance(value, BaseModel):
            raise ConversionError(
                "Value must be a Pydantic model instance", type(value), target_type, value
            )

        if not self._is_pydantic_model(target_type):
            raise ConversionError(
                f"Target type {target_type} is not a Pydantic model",
                type(value),
                target_type,
                value,
            )

        try:
            # Convert source model to dict, then to target model
            intermediate_dict = value.model_dump()
            result = target_type.model_validate(intermediate_dict)  # type: ignore[attr-defined]
            return result  # type: ignore[no-any-return]
        except ValidationError as e:
            error_details = []
            for error in e.errors():
                loc = ".".join(str(x) for x in error["loc"])
                error_details.append(f"{loc}: {error['msg']}")

            raise ConversionError(
                f"Failed to convert between Pydantic models: {'; '.join(error_details)}",
                type(value),
                target_type,
                value,
            ) from e
        except Exception as e:
            raise ConversionError(
                f"Unexpected error during Pydantic model conversion: {str(e)}",
                type(value),
                target_type,
                value,
            ) from e

    def _is_pydantic_model(self, type_: type) -> bool:
        """Check if type is a Pydantic model."""
        try:
            return issubclass(type_, BaseModel)
        except TypeError:
            return False
