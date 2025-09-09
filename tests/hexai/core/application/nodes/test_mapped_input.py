"""Tests for automatic input mapping with Pydantic."""

import pytest
from pydantic import BaseModel

from hexai.core.application.nodes.mapped_input import (
    AutoMappedInput,
    FieldMappingRegistry,
    MappedInput,
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
