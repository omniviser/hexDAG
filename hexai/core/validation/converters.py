"""Type converter interface for the unified validation framework.

This module provides a flexible type conversion system that allows custom converters to be
registered for specific type conversions.
"""

from abc import ABC, abstractmethod
from typing import Any


class ConversionError(Exception):
    """Exception raised when type conversion fails."""

    def __init__(self, message: str, source_type: type, target_type: type, value: Any):
        self.message = message
        self.source_type = source_type
        self.target_type = target_type
        self.value = value
        super().__init__(f"{message} (converting {source_type.__name__} to {target_type.__name__})")


class TypeConverter(ABC):
    """Abstract base class for type converters.

    Type converters are responsible for converting values from one type to another. They can be
    registered with the ConversionRegistry to be used automatically by the validation framework.
    """

    @abstractmethod
    def can_convert(self, source_type: type, target_type: type) -> bool:
        """Check if this converter can handle the specified type conversion.

        Args
        ----
            source_type: The type of the input value
            target_type: The type to convert to

        Returns
        -------
            True if this converter can handle the conversion, False otherwise
        """
        pass

    @abstractmethod
    def convert(self, value: Any, target_type: type) -> Any:
        """Convert a value to the target type.

        Args
        ----
            value: The value to convert
            target_type: The type to convert to

        Returns
        -------
            The converted value

        Raises
        ------
            ConversionError: If the conversion fails
        """
        pass


class ConversionRegistry:
    """Registry for type converters.

    This class manages the registration and lookup of type converters. Converters can be registered
    for specific type pairs or for general conversion patterns.
    """

    def __init__(self) -> None:
        self._converters: list[TypeConverter] = []
        self._specific_converters: dict[tuple[type, type], TypeConverter] = {}

    def register(self, converter: TypeConverter) -> None:
        """Register a type converter.

        Args
        ----
            converter: The converter to register
        """
        self._converters.append(converter)

    def register_specific(
        self, source_type: type, target_type: type, converter: TypeConverter
    ) -> None:
        """Register a converter for a specific type pair.

        Args
        ----
            source_type: The source type
            target_type: The target type
            converter: The converter to use for this specific conversion
        """
        self._specific_converters[(source_type, target_type)] = converter

    def get_converter(self, source_type: type, target_type: type) -> TypeConverter | None:
        """Get a converter for the specified type conversion.

        Args
        ----
            source_type: The source type
            target_type: The target type

        Returns
        -------
            The converter if found, None otherwise
        """
        # Check for specific converter first
        specific_key = (source_type, target_type)
        if specific_key in self._specific_converters:
            return self._specific_converters[specific_key]

        # Check general converters
        for converter in self._converters:
            if converter.can_convert(source_type, target_type):
                return converter

        return None

    def can_convert(self, source_type: type, target_type: type) -> bool:
        """Check if a converter exists for the specified type conversion.

        Args
        ----
            source_type: The source type
            target_type: The target type

        Returns
        -------
            True if a converter exists, False otherwise
        """
        return self.get_converter(source_type, target_type) is not None

    def convert(self, value: Any, target_type: type) -> Any:
        """Convert a value to the target type using registered converters.

        Args
        ----
            value: The value to convert
            target_type: The type to convert to

        Returns
        -------
            The converted value

        Raises
        ------
            ConversionError: If no converter is found or conversion fails
        """
        source_type = type(value)

        # If types are already compatible, no conversion needed
        if isinstance(value, target_type):
            return value

        converter = self.get_converter(source_type, target_type)
        if converter is None:
            raise ConversionError(
                f"No converter found for {source_type.__name__} to {target_type.__name__}",
                source_type,
                target_type,
                value,
            )

        try:
            return converter.convert(value, target_type)
        except Exception as e:
            if isinstance(e, ConversionError):
                raise
            raise ConversionError(
                f"Conversion failed: {str(e)}", source_type, target_type, value
            ) from e


# Global registry instance
_registry: ConversionRegistry = ConversionRegistry()


def get_registry() -> ConversionRegistry:
    """Get the global conversion registry."""
    return _registry


def register_converter(converter: TypeConverter) -> None:
    """Register a converter with the global registry."""
    _registry.register(converter)


def register_specific_converter(
    source_type: type, target_type: type, converter: TypeConverter
) -> None:
    """Register a specific converter with the global registry."""
    _registry.register_specific(source_type, target_type, converter)


def can_convert(source_type: type, target_type: type) -> bool:
    """Check if a converter exists for the specified type conversion."""
    return _registry.can_convert(source_type, target_type)


def convert(value: Any, target_type: type) -> Any:
    """Convert a value to the target type using the global registry."""
    return _registry.convert(value, target_type)
