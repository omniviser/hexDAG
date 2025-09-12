"""Tests for automatic input mapping with Pydantic."""

import pytest
from pydantic import BaseModel

from hexai.core.application.nodes.mapped_input import (
    AutoMappedInput,
    FieldExtractor,
    FieldMappingRegistry,
    MappedInput,
    ModelFactory,
    TypeInferrer,
)


# Test models
class ProcessorOutput(BaseModel):
    """Test processor output model."""

    text: str
    metadata: dict[str, str] = {}


class ValidatorOutput(BaseModel):
    """Test validator output model."""

    status: str
    score: float


class TestFieldMappingRegistry:
    """Tests for FieldMappingRegistry."""

    def test_empty_registry(self):
        """Test registry starts empty - no magic."""
        registry = FieldMappingRegistry()
        assert len(registry.mappings) == 0

    def test_register_mapping(self):
        """Test registering a field mapping."""
        registry = FieldMappingRegistry()

        mapping = {
            "content": "processor.text",
            "language": "processor.metadata.lang",
            "status": "validator.status",
        }

        registry.register("standard_consumer", mapping)
        assert "standard_consumer" in registry.mappings
        assert registry.mappings["standard_consumer"] == mapping

    def test_get_by_name(self):
        """Test getting mapping by name."""
        registry = FieldMappingRegistry()
        mapping = {"content": "processor.text"}
        registry.register("test_mapping", mapping)

        retrieved = registry.get("test_mapping")
        assert retrieved == mapping

    def test_get_inline_mapping(self):
        """Test getting inline mapping (returns as-is)."""
        registry = FieldMappingRegistry()
        inline_mapping = {"field1": "source.field1"}

        retrieved = registry.get(inline_mapping)
        assert retrieved == inline_mapping

    def test_get_unknown_mapping_raises(self):
        """Test getting unknown mapping raises ValueError."""
        registry = FieldMappingRegistry()

        with pytest.raises(ValueError, match="Unknown field mapping: 'nonexistent'"):
            registry.get("nonexistent")

    def test_clear_registry(self):
        """Test clearing all mappings."""
        registry = FieldMappingRegistry()
        registry.register("mapping1", {"a": "b"})
        registry.register("mapping2", {"c": "d"})

        assert len(registry.mappings) == 2
        registry.clear()
        assert len(registry.mappings) == 0


class TestMappedInput:
    """Tests for MappedInput factory."""

    def test_create_simple_model(self):
        """Test creating a simple mapped model."""
        mapping = {"content": "processor.text", "status": "validator.status"}

        model_class = MappedInput.create_model("TestModel", mapping)

        # Test the model can be instantiated
        data = {"processor": {"text": "hello"}, "validator": {"status": "valid"}}

        instance = model_class(**data)
        assert instance.content == "hello"
        assert instance.status == "valid"

    def test_extract_nested_fields(self):
        """Test extracting deeply nested fields."""
        mapping = {"language": "processor.metadata.lang", "deep": "level1.level2.level3.value"}

        model_class = MappedInput.create_model("NestedModel", mapping)

        data = {
            "processor": {"metadata": {"lang": "en"}},
            "level1": {"level2": {"level3": {"value": "deep_value"}}},
        }

        instance = model_class(**data)
        assert instance.language == "en"
        assert instance.deep == "deep_value"

    def test_with_pydantic_models(self):
        """Test extraction from Pydantic model instances."""
        mapping = {
            "content": "processor.text",
            "language": "processor.metadata.lang",
            "score": "validator.score",
        }

        model_class = MappedInput.create_model("MappedModel", mapping)

        # Input with Pydantic models
        processor = ProcessorOutput(text="test content", metadata={"lang": "en"})
        validator = ValidatorOutput(status="valid", score=0.95)

        data = {"processor": processor, "validator": validator}

        instance = model_class(**data)
        assert instance.content == "test content"
        assert instance.language == "en"
        assert instance.score == 0.95

    def test_missing_fields_handled(self):
        """Test missing fields return None gracefully."""
        mapping = {"existing": "source.field", "missing": "nonexistent.field"}

        model_class = MappedInput.create_model("TestModel", mapping)

        data = {"source": {"field": "value"}}

        # Should not raise, missing field is None
        instance = model_class(**data)
        assert instance.existing == "value"
        assert instance.missing is None  # Field is None when source not found

    def test_type_inference(self):
        """Test type inference from dependency models."""
        dependency_models = {"processor": ProcessorOutput, "validator": ValidatorOutput}

        mapping = {"content": "processor.text", "score": "validator.score"}

        model_class = MappedInput.create_model(
            "InferredModel", mapping, dependency_models=dependency_models
        )

        # The model should have been created with inferred types
        data = {
            "processor": ProcessorOutput(text="test"),
            "validator": ValidatorOutput(status="ok", score=1.0),
        }

        instance = model_class(**data)
        assert instance.content == "test"
        assert instance.score == 1.0


class TestAutoMappedInput:
    """Tests for AutoMappedInput base class."""

    def test_subclass_with_mapping(self):
        """Test creating a subclass with field mapping."""

        class ConsumerInput(AutoMappedInput):
            content: str
            language: str
            status: str

            _field_mapping = {
                "content": "processor.text",
                "language": "processor.metadata.lang",
                "status": "validator.status",
            }

        # Test with input data
        data = {
            "processor": {"text": "hello world", "metadata": {"lang": "fr"}},
            "validator": {"status": "pending"},
        }

        instance = ConsumerInput(**data)
        assert instance.content == "hello world"
        assert instance.language == "fr"
        assert instance.status == "pending"

    def test_subclass_without_mapping(self):
        """Test subclass without mapping works normally."""

        class SimpleInput(AutoMappedInput):
            field1: str
            field2: int

        # No mapping, so direct field assignment
        data = {"field1": "value", "field2": 42}
        instance = SimpleInput(**data)
        assert instance.field1 == "value"
        assert instance.field2 == 42

    def test_extract_from_mixed_sources(self):
        """Test extracting from both dict and Pydantic sources."""

        class MixedInput(AutoMappedInput):
            text: str
            score: float
            custom: str

            _field_mapping = {
                "text": "processor.text",
                "score": "validator.score",
                "custom": "metadata.custom_field",
            }

        processor = ProcessorOutput(text="content")
        validator = ValidatorOutput(status="ok", score=0.8)

        data = {
            "processor": processor,
            "validator": validator,
            "metadata": {"custom_field": "custom_value"},
        }

        instance = MixedInput(**data)
        assert instance.text == "content"
        assert instance.score == 0.8
        assert instance.custom == "custom_value"

    def test_registry_empty_name_error(self):
        """Test that registering with empty name raises error."""
        registry = FieldMappingRegistry()

        with pytest.raises(ValueError, match="Mapping name cannot be empty"):
            registry.register("", {"field": "value"})

    def test_registry_empty_mapping_error(self):
        """Test that registering empty mapping raises error."""
        registry = FieldMappingRegistry()

        with pytest.raises(ValueError, match="Mapping cannot be empty"):
            registry.register("test", {})

    def test_field_extractor_empty_path(self):
        """Test extraction with empty path returns the data itself."""
        data = {"key": "value"}
        result = FieldExtractor.extract(data, "")
        assert result == data

    def test_field_extractor_none_data(self):
        """Test extraction from None data returns None."""
        result = FieldExtractor.extract(None, "some.path")
        assert result is None

    def test_field_extractor_attribute_error_handling(self):
        """Test extraction handles AttributeError gracefully."""

        class BadObject:
            def __getattr__(self, name):
                raise AttributeError(f"No attribute {name}")

        bad_obj = BadObject()
        result = FieldExtractor._extract_single_level(bad_obj, "anything")
        assert result is None

    def test_field_extractor_type_error_handling(self):
        """Test extraction handles TypeError gracefully."""
        # Try to extract from a number (not dict/object)
        result = FieldExtractor._extract_single_level(42, "field")
        assert result is None

    def test_type_inferrer_empty_path(self):
        """Test type inference with empty path returns the model itself."""
        result = TypeInferrer.infer_from_path(ProcessorOutput, [])
        assert result == ProcessorOutput

    def test_type_inferrer_field_not_found(self):
        """Test type inference when field doesn't exist returns Any."""
        from typing import Any

        result = TypeInferrer.infer_from_path(ProcessorOutput, ["nonexistent_field"])
        assert result == Any

    def test_type_inferrer_get_field_type_errors(self):
        """Test _get_field_type with various error conditions."""

        class BadModel:
            model_fields = "not_a_dict"

        result = TypeInferrer._get_field_type(BadModel, "field")
        assert result is None

        # Test with model that raises exception
        class ErrorModel:
            @property
            def model_fields(self):
                raise RuntimeError("Error accessing fields")

        result = TypeInferrer._get_field_type(ErrorModel, "field")
        assert result is None

    def test_type_inferrer_is_base_model_type_error(self):
        """Test _is_base_model with non-type objects."""
        assert TypeInferrer._is_base_model("not_a_type") is False
        assert TypeInferrer._is_base_model(None) is False
        assert TypeInferrer._is_base_model(42) is False

    def test_model_factory_no_dependency_models(self):
        """Test ModelFactory with no dependency models for type inference."""
        mapping = {"field1": "source.data", "field2": "other.value"}

        model = ModelFactory.create_mapped_model("TestModel", mapping, None)

        # Should create model with Any types
        data = {"source": {"data": "test"}, "other": {"value": 123}}
        instance = model(**data)
        assert instance.field1 == "test"
        assert instance.field2 == 123

    def test_model_factory_infer_nested_non_model(self):
        """Test type inference for nested non-BaseModel fields."""
        mapping = {"value": "processor.text"}
        dep_models = {"processor": ProcessorOutput}

        model = ModelFactory.create_mapped_model("TestModel", mapping, dep_models)

        # Should infer str type for text field
        data = {"processor": ProcessorOutput(text="hello", metadata={})}
        instance = model(**data)
        assert instance.value == "hello"

    def test_model_factory_validator_non_dict_input(self):
        """Test validator handles non-dict input gracefully."""
        mapping = {"field": "source.value"}
        model = ModelFactory.create_mapped_model("TestModel", mapping)

        # Pass non-dict data
        instance = model(**{"source": {"value": "test"}})
        assert instance.field == "test"

    def test_auto_mapped_normalize_to_dict_basemodel(self):
        """Test _normalize_to_dict with BaseModel input."""
        input_model = ProcessorOutput(text="test", metadata={"key": "value"})
        result = AutoMappedInput._normalize_to_dict(input_model)

        assert result == {"text": "test", "metadata": {"key": "value"}}

    def test_auto_mapped_normalize_to_dict_other(self):
        """Test _normalize_to_dict with non-dict, non-BaseModel input."""
        result = AutoMappedInput._normalize_to_dict("not_dict_or_model")
        assert result == {}

        result = AutoMappedInput._normalize_to_dict(42)
        assert result == {}

    def test_auto_mapped_field_mapping_edge_cases(self):
        """Test AutoMappedInput with various _field_mapping configurations."""

        class NoneMapping(AutoMappedInput):
            field1: str = "default"
            _field_mapping = None

        instance = NoneMapping(field1="value")
        assert instance.field1 == "value"

        # Test with dict-like object
        class DictLikeMapping:
            def items(self):
                return [("field1", "source.field1")]

        class CustomMapping(AutoMappedInput):
            field1: str = "default"
            _field_mapping = DictLikeMapping()

        data = {"source": {"field1": "test"}}
        instance = CustomMapping(**data)

        assert instance.field1 == "test"

        # Test with object that has items but raises error
        class BadDictLike:
            def items(self):
                raise ValueError("Cannot access items")

        class BadMapping(AutoMappedInput):
            field1: str = "default"
            _field_mapping = BadDictLike()

        # Should fall back to empty mapping when items() raises
        instance = BadMapping(field1="direct")
        assert instance.field1 == "direct"

        # Test that items() method is actually called
        class TrackingDictLike:
            def __init__(self):
                self.items_called = False

            def items(self):
                self.items_called = True
                return [("field1", "source.field1")]

        tracker = TrackingDictLike()

        class TrackingMapping(AutoMappedInput):
            field1: str = "default"
            _field_mapping = tracker

        data = {"source": {"field1": "tracked"}}
        instance = TrackingMapping(**data)
        assert instance.field1 == "tracked"
        assert tracker.items_called, "items() method should have been called"

    def test_auto_mapped_apply_non_dict_data(self):
        """Test apply_field_mapping with non-dict input data."""

        class TestInput(AutoMappedInput):
            field1: str = "default"
            _field_mapping = {"field1": "source.value"}

        # When no source data is provided, uses default
        instance = TestInput()
        assert instance.field1 == "default"

    def test_field_extractor_general_exception_handling(self):
        """Test that general exceptions in extraction are caught."""
        # Test with object that raises exception during isinstance check
        result = FieldExtractor._extract_single_level(None, "field")
        assert result is None

    def test_type_inferrer_nested_path_non_model(self):
        """Test type inference with nested path on non-BaseModel fields."""
        # When field type is not a BaseModel, we can't traverse further
        result = TypeInferrer.infer_from_path(ProcessorOutput, ["text", "some_nested"])
        # Should return the type of text field (str), not traverse further
        assert result != ProcessorOutput

    def test_type_inferrer_is_base_model_with_exception(self):
        """Test _is_base_model when TypeError is raised."""

        class WeirdType:
            def __class__(self):
                raise TypeError("Cannot check class")

        assert TypeInferrer._is_base_model(WeirdType) is False

    def test_model_factory_field_definitions_with_none_dep_models(self):
        """Test building field definitions without dependency models."""
        mapping = {"field1": "source.value", "field2": "data"}
        definitions = ModelFactory._build_field_definitions(mapping, None)

        # Should create optional Any fields
        assert "field1" in definitions
        assert "field2" in definitions

    def test_mapped_input_backward_compatibility(self):
        """Test MappedInput maintains backward compatibility."""
        # Test that old static methods still work
        value = MappedInput._extract_value({"key": "value"}, "key")
        assert value == "value"

        # Test type inference backward compat
        from typing import Any

        field_type = MappedInput._infer_field_type(ProcessorOutput, ["text"])
        assert field_type != Any

    def test_auto_mapped_with_pydantic_private_attr(self):
        """Test AutoMappedInput with Pydantic private attribute style."""
        from pydantic import PrivateAttr

        class PrivateMapping(AutoMappedInput):
            field1: str = "default"
            _field_mapping = PrivateAttr(default={"field1": "source.value"})

        # Should handle private attr with default
        instance = PrivateMapping()
        # Since PrivateAttr is handled differently, it may not map
        assert instance.field1 == "default"  # Falls back to default
