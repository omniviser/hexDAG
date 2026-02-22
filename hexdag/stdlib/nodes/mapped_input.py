"""Automatic input mapping using Pydantic models."""

from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel, Field, create_model, model_validator

from hexdag.kernel.exceptions import ResourceNotFoundError, ValidationError
from hexdag.kernel.protocols import DictConvertible, is_dict_convertible, is_schema_type


class FieldMappingRegistry:
    """Registry for common field mappings - empty by default, no magic."""

    def __init__(self) -> None:
        """Initialize empty registry - users must define their mappings."""
        self.mappings: dict[str, dict[str, str]] = {}

    def register(self, name: str, mapping: dict[str, str]) -> None:
        """Register a reusable field mapping.

        Args
        ----
            name: Name for the mapping pattern
            mapping: dict of {target_field: "source.path"}

        Raises
        ------
        ValidationError
            If the mapping name is empty
        """
        if not name:
            raise ValidationError("name", "cannot be empty")
        if not mapping:
            raise ValidationError("mapping", "cannot be empty")
        self.mappings[name] = mapping

    def get(self, name_or_mapping: str | dict[str, str]) -> dict[str, str]:
        """Get mapping by name or return inline mapping.

        Args
        ----
            name_or_mapping: Either a string name or inline mapping dict

        Returns
        -------
        dict[str, str]
            The resolved mapping dictionary

        Raises
        ------
        ResourceNotFoundError
            If the mapping name is not found in registry
        """
        if isinstance(name_or_mapping, str):
            if name_or_mapping not in self.mappings:
                available = list(self.mappings.keys()) if self.mappings else []
                raise ResourceNotFoundError("field mapping", name_or_mapping, available)
            return self.mappings[name_or_mapping]
        return name_or_mapping

    def clear(self) -> None:
        """Clear all registered mappings."""
        self.mappings.clear()


class FieldExtractor:
    """Handles extraction of values from nested data structures."""

    @staticmethod
    def extract(data: dict[Any, Any] | DictConvertible, path: str) -> Any:
        """Extract value from nested data structure using dot notation path.

        Args
        ----
            data: The data structure to extract from
            path: Dot-separated path to the value (e.g., "user.profile.name")

        Returns
        -------
            The extracted value or None if not found

        """
        if not path:
            return data

        parts = path.split(".")
        current: Any = data

        for part in parts:
            current = FieldExtractor._extract_single_level(current, part)
            if current is None:
                break

        return current

    @staticmethod
    def _extract_single_level(data: Any, key: str) -> Any:
        """Extract a single level from the data.

        Args
        ----
            data: Current data object
            key: The key/attribute to extract

        Returns
        -------
            The value at the key or None if not found

        """
        if data is None:
            return None

        if isinstance(data, dict):
            return data.get(key)

        if is_dict_convertible(data):
            return getattr(data, key, None)

        # Try generic attribute access for other objects
        try:
            return getattr(data, key, None)
        except (AttributeError, TypeError):
            return None


class TypeInferrer:
    """Handles type inference from Pydantic models."""

    @staticmethod
    def infer_from_path(model: type[BaseModel], field_path: list[str]) -> type[Any]:
        """Infer field type from a Pydantic model and field path.

        Args
        ----
            model: The Pydantic model class
            field_path: list of field names forming the path

        Returns
        -------
            The inferred type or Any if inference fails

        """
        if not field_path:
            return model

        field_name = field_path[0]
        field_type = TypeInferrer._get_field_type(model, field_name)

        if field_type is None:
            return cast("type[Any]", Any)

        # Recurse for nested paths
        if len(field_path) > 1 and TypeInferrer._is_base_model(field_type):
            return TypeInferrer.infer_from_path(field_type, field_path[1:])

        return field_type

    @staticmethod
    def _get_field_type(model: type[BaseModel], field_name: str) -> type[Any] | None:
        """Get the type of a specific field from a model.

        Args
        ----
            model: The Pydantic model class
            field_name: Name of the field

        Returns
        -------
            The field type or None if not found

        """
        try:
            # Pydantic v2 approach - use protocol check
            if is_schema_type(model):
                model_fields = getattr(model, "model_fields", {})
                if field_name in model_fields:
                    annotation: type[Any] = model_fields[field_name].annotation
                    return annotation
        except (AttributeError, TypeError, KeyError):
            # Field not found or error accessing it
            pass

        return None

    @staticmethod
    def _is_base_model(field_type: Any) -> bool:
        """Check if a type is a BaseModel subclass.

        Args
        ----
            field_type: The type to check

        Returns
        -------
            True if the type is a BaseModel subclass

        """
        return is_schema_type(field_type)


class ModelFactory:
    """Factory for creating mapped Pydantic models."""

    @staticmethod
    def create_mapped_model(
        name: str,
        mapping: dict[str, str],
        dependency_models: dict[str, type[BaseModel]] | None = None,
    ) -> type[BaseModel]:
        """Create a Pydantic model with automatic field mapping.

        Args
        ----
            name: Name for the generated model
            mapping: Field mapping {target_field: "source.field.path"}
            dependency_models: Optional dict of {dep_name: OutputModel} for type inference

        Returns
        -------
            Pydantic model class with automatic field extraction

        """
        field_definitions = ModelFactory._build_field_definitions(mapping, dependency_models)

        validator = ModelFactory._create_validator(mapping)

        model: type[BaseModel] = create_model(
            name,
            __validators__={"extract_mapped_fields": validator},
            **field_definitions,
        )

        model._field_mapping = mapping  # type: ignore[attr-defined]

        return model

    @staticmethod
    def _build_field_definitions(
        mapping: dict[str, str], dependency_models: dict[str, type[BaseModel]] | None
    ) -> dict[str, Any]:
        """Build field definitions with type inference.

        Args
        ----
            mapping: Field mapping dictionary
            dependency_models: Optional dependency models for type inference

        Returns
        -------
            dictionary of field definitions for create_model

        """
        definitions: dict[str, Any] = {}

        for target_field, source_path in mapping.items():
            field_type = ModelFactory._infer_field_type(source_path, dependency_models)
            # Make fields optional by default since mapping might not provide them
            definitions[target_field] = (field_type | None, Field(default=None))

        return definitions

    @staticmethod
    def _infer_field_type(
        source_path: str, dependency_models: dict[str, type[BaseModel]] | None
    ) -> type[Any]:
        """Infer type for a field from source path.

        Args
        ----
            source_path: The source path string
            dependency_models: Optional dependency models

        Returns
        -------
            The inferred type or Any

        """
        if not dependency_models or "." not in source_path:
            return cast("type[Any]", Any)

        parts = source_path.split(".")
        dep_name = parts[0]

        if dep_name not in dependency_models:
            return cast("type[Any]", Any)

        return TypeInferrer.infer_from_path(dependency_models[dep_name], parts[1:])

    @staticmethod
    def _create_validator(mapping: dict[str, str]) -> Any:
        """Create the field extraction validator.

        Args
        ----
            mapping: Field mapping dictionary

        Returns
        -------
            A Pydantic validator function

        """

        def extract_mapped_fields(data: Any) -> dict[str, Any]:
            """Extract fields from nested structure based on mapping.

            This validator handles two scenarios:
            1. Data pre-processed by ExecutionCoordinator._apply_input_mapping
               - Target fields already exist in data with resolved values
               - Just pass through the pre-resolved data
            2. Raw dependency data that needs extraction
               - Use FieldExtractor to extract values from nested structures
            """
            if not isinstance(data, dict):
                return {}

            result: dict[str, Any] = {}
            target_fields = set(mapping.keys())

            # Check if data was pre-processed by ExecutionCoordinator
            # Pre-processed data has the target field names as keys (not source paths)
            data_keys = set(data.keys())
            if target_fields <= data_keys:
                # Data already has all target fields - it was pre-processed
                # Just extract the target fields directly
                for target_field in target_fields:
                    result[target_field] = data.get(target_field)
                return result

            # Data needs extraction using the mapping paths
            for target_field, source_path in mapping.items():
                # Skip $input paths - these should have been resolved by ExecutionCoordinator
                # If we get here with $input paths, the data wasn't pre-processed
                if source_path.startswith("$input"):
                    # Try to find the target field directly in data
                    if target_field in data:
                        result[target_field] = data[target_field]
                    continue

                value = FieldExtractor.extract(data, source_path)
                if value is not None:
                    result[target_field] = value

            return result

        return model_validator(mode="before")(extract_mapped_fields)


class MappedInput:
    """Factory for creating auto-mapped Pydantic input models.

    This class provides a simple API for creating Pydantic models
    that automatically map fields from nested input structures.

    Example
    -------
        ConsumerInput = MappedInput.create_model(
            "ConsumerInput",
            {
                "content": "processor.text",
                "language": "processor.metadata.lang",
                "status": "validator.status"
            }
        )

    """

    @staticmethod
    def create_model(
        name: str,
        mapping: dict[str, str],
        dependency_models: dict[str, type[BaseModel]] | None = None,
    ) -> type[BaseModel]:
        """Create a Pydantic model with automatic field mapping.

        Args:
        ----
            name: Name for the generated model
            mapping: Field mapping {target_field: "source.field.path"}
            dependency_models: Optional dict of {dep_name: OutputModel} for type inference

        Returns
        -------
            Pydantic model class with automatic field extraction

        Example
        -------
            ConsumerInput = MappedInput.create_model(
                "ConsumerInput",
                {
                    "content": "processor.text",
                    "language": "processor.metadata.lang",
                    "status": "validator.status"
                }
            )

        """
        return ModelFactory.create_mapped_model(name, mapping, dependency_models)

    # Maintain backward compatibility
    _extract_value = staticmethod(FieldExtractor.extract)
    _infer_field_type = staticmethod(TypeInferrer.infer_from_path)


class AutoMappedInput(BaseModel):
    """Base class for models with automatic field mapping.

    Users can subclass this to create models with field mapping.

    Example
    -------
        class ConsumerInput(AutoMappedInput):
            content: str
            language: str
            status: str

            _field_mapping = {
                "content": "processor.text",
                "language": "processor.metadata.lang",
                "status": "validator.status"
            }

    """

    @model_validator(mode="before")
    @classmethod
    def apply_field_mapping(cls: type[AutoMappedInput], data: Any) -> dict[str, Any]:
        """Automatically apply field mapping before validation.

        Args
        ----
            cls: The class being instantiated
            data: Input data to be mapped

        Returns
        -------
            Mapped data dictionary

        """
        field_mapping = cls._get_field_mapping()

        if not field_mapping:
            return cls._normalize_to_dict(data)

        if not isinstance(data, dict):
            return {}

        result: dict[str, Any] = {}
        for target_field, source_path in field_mapping.items():
            value = FieldExtractor.extract(data, source_path)
            if value is not None:
                result[target_field] = value

        return result

    @classmethod
    def _get_field_mapping(cls) -> dict[str, str]:
        """Get the field mapping from the class.

        Returns
        -------
            The field mapping dictionary or empty dict

        """
        if not hasattr(cls, "_field_mapping"):
            return {}

        mapping_attr = getattr(cls, "_field_mapping", None)

        if mapping_attr is None:
            return {}

        # Direct dict assignment on the class
        if isinstance(mapping_attr, dict):
            return mapping_attr

        # Pydantic private attr with default
        if hasattr(mapping_attr, "default"):
            default_value = getattr(mapping_attr, "default", {})
            if isinstance(default_value, dict):
                return default_value
            if hasattr(default_value, "items"):
                try:
                    return dict(default_value.items())
                except (TypeError, ValueError):
                    pass
            return {}

        # Try to use it directly if it's dict-like
        if hasattr(mapping_attr, "items"):
            try:
                # Call items() method to get the key-value pairs
                return dict(mapping_attr.items())
            except (TypeError, ValueError, AttributeError):
                pass

        return {}

    @staticmethod
    def _normalize_to_dict(data: Any) -> dict[str, Any]:
        """Normalize data to a dictionary.

        Args
        ----
            data: Input data

        Returns
        -------
            dictionary representation of the data

        """
        if isinstance(data, dict):
            return data
        if is_dict_convertible(data):
            result: dict[str, Any] = data.model_dump()
            return result
        return {}
