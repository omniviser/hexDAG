"""Test cases for BaseNodeFactory class."""

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from hexdag.kernel.domain.dag import NodeSpec
from hexdag.stdlib.nodes.base_node_factory import BaseNodeFactory


class MockNodeFactory(BaseNodeFactory):
    """Mock implementation of BaseNodeFactory for testing."""

    def __call__(self, name: str, **kwargs: Any) -> NodeSpec:
        """Mock implementation."""

        async def mock_fn(input_data: Any) -> str:
            return f"Mock output for {name}"

        return NodeSpec(
            name=name,
            fn=mock_fn,
            in_model=None,
            out_model=None,
            deps=set(kwargs.get("deps", [])),
            params=kwargs,
        )


class TestBaseNodeFactory:
    """Test cases for BaseNodeFactory class."""

    @pytest.fixture
    def factory(self):
        """Fixture for MockNodeFactory."""
        return MockNodeFactory()

    def test_create_pydantic_model_from_dict(self, factory):
        """Test creating Pydantic model from dict schema."""
        schema = {"name": str, "age": int}
        model = factory.create_pydantic_model("TestModel", schema)

        assert model is not None
        assert issubclass(model, BaseModel)

        # Test instance creation
        instance = model(name="John", age=30)
        assert instance.name == "John"
        assert instance.age == 30

    def test_create_pydantic_model_from_basemodel(self, factory):
        """Test creating model from existing BaseModel."""

        class ExistingModel(BaseModel):
            value: str

        model = factory.create_pydantic_model("TestModel", ExistingModel)
        assert model is ExistingModel

    def test_create_pydantic_model_none_schema(self, factory):
        """Test creating model with None schema."""
        model = factory.create_pydantic_model("TestModel", None)
        assert model is None

    def test_create_pydantic_model_from_primitive_type(self, factory):
        """Test creating model from primitive type."""
        model = factory.create_pydantic_model("TestModel", str)

        assert model is not None
        assert issubclass(model, BaseModel)

        # Test instance creation
        instance = model(value="test")
        assert instance.value == "test"

    def test_emit_methods_removed(self, factory):
        """Test that event emission methods have been removed."""
        # The BaseNodeFactory no longer handles events
        # Events are now managed by the orchestrator via ObserverManager
        assert not hasattr(factory, "emit_node_started")
        assert not hasattr(factory, "emit_node_completed")
        assert not hasattr(factory, "emit_node_failed")

    def test_create_node_with_mapping(self, factory):
        """Test creating node with input mapping."""

        def test_fn():
            return "test"

        input_schema = {"field1": str}
        input_mapping = {"field1": "source.field1"}

        node_spec = factory.create_node_with_mapping(
            name="test_node",
            wrapped_fn=test_fn,
            input_schema=input_schema,
            output_schema=None,
            deps=["dep1"],
            input_mapping=input_mapping,
        )

        assert node_spec.name == "test_node"
        assert node_spec.fn == test_fn
        assert "input_mapping" in node_spec.params
        assert node_spec.params["input_mapping"] == input_mapping


class TestNullableSchema:
    """Tests for nullable type support in create_pydantic_model."""

    @pytest.fixture
    def factory(self):
        return MockNodeFactory()

    def test_nullable_accepts_none(self, factory):
        model = factory.create_pydantic_model("Test", {"name": "str?"})
        assert model(name=None).name is None

    def test_nullable_accepts_value(self, factory):
        model = factory.create_pydantic_model("Test", {"name": "str?"})
        assert model(name="hello").name == "hello"

    def test_nullable_defaults_to_none(self, factory):
        model = factory.create_pydantic_model("Test", {"name": "str?", "age": "int"})
        instance = model(age=30)
        assert instance.name is None
        assert instance.age == 30

    def test_required_rejects_none(self, factory):
        model = factory.create_pydantic_model("Test", {"name": str})
        with pytest.raises(ValidationError):
            model(name=None)

    def test_mixed_required_and_nullable(self, factory):
        model = factory.create_pydantic_model(
            "Test",
            {
                "rate": float,
                "origin": "str?",
                "mc_number": "str?",
            },
        )
        instance = model(rate=2100.0, origin=None, mc_number=None)
        assert instance.rate == 2100.0
        assert instance.origin is None
        assert instance.mc_number is None

    def test_all_nullable_variants(self, factory):
        model = factory.create_pydantic_model(
            "Test",
            {
                "s": "str?",
                "i": "int?",
                "f": "float?",
                "b": "bool?",
                "l": "list?",
                "d": "dict?",
            },
        )
        instance = model()
        for field in ("s", "i", "f", "b", "l", "d"):
            assert getattr(instance, field) is None


class TestPydanticModelCaching:
    """Tests for Pydantic model creation caching."""

    @pytest.fixture
    def factory(self):
        return MockNodeFactory()

    def test_same_schema_returns_cached_model(self, factory):
        """Identical schema dicts return the exact same model class."""
        schema = {"name": str, "age": int}
        model1 = factory.create_pydantic_model("CachedModel", schema)
        model2 = factory.create_pydantic_model("CachedModel", schema)
        assert model1 is model2

    def test_different_schemas_return_different_models(self, factory):
        """Different schemas produce different model classes."""
        model1 = factory.create_pydantic_model("Model1", {"name": str})
        model2 = factory.create_pydantic_model("Model1", {"email": str})
        assert model1 is not model2

    def test_different_names_same_schema_return_different_models(self, factory):
        """Same schema but different model names are cached separately."""
        schema = {"name": str}
        model1 = factory.create_pydantic_model("InputModel", schema)
        model2 = factory.create_pydantic_model("OutputModel", schema)
        assert model1 is not model2

    def test_cached_model_still_works(self, factory):
        """Cached models can still create valid instances."""
        schema = {"name": str, "value": int}
        model = factory.create_pydantic_model("WorkingModel", schema)
        instance = model(name="test", value=42)
        assert instance.name == "test"
        assert instance.value == 42

        # Get from cache and verify it still works
        model2 = factory.create_pydantic_model("WorkingModel", schema)
        instance2 = model2(name="other", value=99)
        assert instance2.name == "other"

    def test_basemodel_passthrough_not_cached(self, factory):
        """Existing BaseModel subclasses are returned as-is, not cached."""

        class MyModel(BaseModel):
            x: int

        result = factory.create_pydantic_model("Ignored", MyModel)
        assert result is MyModel
