"""Test cases for BaseNodeFactory class."""

from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from hexai.core.application.events.events import (
    NodeCompletedEvent,
    NodeFailedEvent,
    NodeStartedEvent,
)
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

    @pytest.fixture
    def mock_event_manager(self):
        """Fixture for mock event manager."""
        event_manager = AsyncMock()
        return event_manager

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

    @pytest.mark.asyncio
    async def test_emit_node_started(self, factory, mock_event_manager):
        """Test emitting node started event."""
        await factory.emit_node_started(
            "test_node", 0, ["dep1", "dep2"], mock_event_manager, {"test": "metadata"}
        )

        mock_event_manager.emit.assert_called_once()
        event = mock_event_manager.emit.call_args[0][0]
        assert isinstance(event, NodeStartedEvent)
        assert event.node_name == "test_node"
        assert event.dependencies == ["dep1", "dep2"]
        assert event.metadata == {"test": "metadata"}

    @pytest.mark.asyncio
    async def test_emit_node_completed(self, factory, mock_event_manager):
        """Test emitting node completed event."""
        await factory.emit_node_completed(
            "test_node", "result", 1.5, 0, mock_event_manager, {"test": "metadata"}
        )

        mock_event_manager.emit.assert_called_once()
        event = mock_event_manager.emit.call_args[0][0]
        assert isinstance(event, NodeCompletedEvent)
        assert event.node_name == "test_node"
        assert event.result == "result"
        assert event.execution_time == 1.5
        assert event.metadata == {"test": "metadata"}

    @pytest.mark.asyncio
    async def test_emit_node_failed(self, factory, mock_event_manager):
        """Test emitting node failed event."""
        error = Exception("test error")
        await factory.emit_node_failed("test_node", error, 0, mock_event_manager)

        mock_event_manager.emit.assert_called_once()
        event = mock_event_manager.emit.call_args[0][0]
        assert isinstance(event, NodeFailedEvent)
        assert event.node_name == "test_node"
        assert event.error == error

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
