"""Basic type converters for the unified validation framework.

This module provides common type converters for string-to-numeric conversions, boolean conversions
from various formats, and list conversions.
"""

from typing import Any

from .converters import ConversionError, TypeConverter


class StringToNumericConverter(TypeConverter):
    """Converter for string to numeric types (int, float)."""

    def can_convert(self, source_type: type, target_type: type) -> bool:
        """Check if we can convert string to numeric type."""
        return source_type == str and target_type in (int, float)

    def convert(self, value: Any, target_type: type) -> int | float:
        """Convert string to int or float."""
        if not isinstance(value, str):
            raise ConversionError("Value must be a string", type(value), target_type, value)

        # Handle empty strings
        if not value.strip():
            raise ConversionError(
                "Cannot convert empty string to numeric type", str, target_type, value
            )

        try:
            if target_type == int:
                # Handle float strings that should become int
                if "." in value:
                    float_val = float(value)
                    if float_val.is_integer():
                        return int(float_val)
                    else:
                        raise ConversionError(
                            f"Cannot convert float {value} to int without loss of precision",
                            str,
                            target_type,
                            value,
                        )
                return int(value)
            elif target_type == float:
                return float(value)
            else:
                raise ConversionError(
                    f"Unsupported target type {target_type}", str, target_type, value
                )
        except ValueError as e:
            raise ConversionError(
                f"Invalid numeric format: {str(e)}", str, target_type, value
            ) from e


class BooleanConverter(TypeConverter):
    """Converter for various formats to boolean."""

    # Common boolean representations
    TRUE_VALUES = {"true", "yes", "y", "1", "on", "enabled", "active", "ok"}
    FALSE_VALUES = {"false", "no", "n", "0", "off", "disabled", "inactive", "none", "null"}

    def can_convert(self, source_type: type, target_type: type) -> bool:
        """Check if we can convert to boolean."""
        return target_type == bool and source_type in (str, int, float)

    def convert(self, value: Any, target_type: type) -> bool:
        """Convert value to boolean."""
        if target_type != bool:
            raise ConversionError("Target type must be bool", type(value), target_type, value)

        # Handle string conversion
        if isinstance(value, str):
            normalized = value.lower().strip()

            if normalized in self.TRUE_VALUES:
                return True
            elif normalized in self.FALSE_VALUES:
                return False
            else:
                raise ConversionError(
                    f"Cannot convert string '{value}' to boolean. "
                    f"Valid true values: {self.TRUE_VALUES}. "
                    f"Valid false values: {self.FALSE_VALUES}",
                    str,
                    bool,
                    value,
                )

        # Handle numeric conversion
        elif isinstance(value, (int, float)):
            if value == 0 or value == 0.0:
                return False
            elif value == 1 or value == 1.0:
                return True
            else:
                raise ConversionError(
                    f"Cannot convert numeric value {value} to boolean. "
                    "Only 0/0.0 (False) and 1/1.0 (True) are supported",
                    type(value),
                    bool,
                    value,
                )

        else:
            raise ConversionError(
                f"Cannot convert {type(value).__name__} to boolean", type(value), bool, value
            )
