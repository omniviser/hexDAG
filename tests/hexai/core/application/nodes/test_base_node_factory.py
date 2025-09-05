"""Test cases for BaseNodeFactory class."""

from typing import Any

import pytest
from pydantic import BaseModel

from hexai.core.application.nodes.base_node_factory import BaseNodeFactory
from hexai.core.domain.dag import NodeSpec


class MockNodeFactory(BaseNodeFactory):
    """Mock implementation of BaseNodeFactory for testing."""

    def __call__(self, name: str, **kwargs: Any) -> NodeSpec:
        """Mock implementation."""

        async def mock_fn(input_data: Any) -> str:
            return f"Mock output for {name}"

        return NodeSpec(
            name=name,
            fn=mock_fn,
            in_type=dict,
            out_type=str,
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
