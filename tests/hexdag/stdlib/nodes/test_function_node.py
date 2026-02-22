"""Tests for FunctionNode factory.

This module tests the FunctionNode factory, including the unpack_input feature
that allows functions to receive mapped fields as individual kwargs.
"""

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from hexdag.stdlib.nodes.function_node import FunctionNode


class TestFunctionNodeCreation:
    """Tests for FunctionNode basic creation."""

    def test_creates_node_spec_with_callable(self) -> None:
        """FunctionNode creates NodeSpec from callable."""
        factory = FunctionNode()

        def my_func(input_data: dict[str, Any]) -> dict[str, Any]:
            return {"result": input_data}

        node = factory(name="test_node", fn=my_func)

        assert node.name == "test_node"
        assert node.fn is not None
        assert node.deps == frozenset()

    def test_creates_node_spec_with_dependencies(self) -> None:
        """FunctionNode creates NodeSpec with dependencies."""
        factory = FunctionNode()

        def my_func(input_data: dict[str, Any]) -> dict[str, Any]:
            return input_data

        node = factory(name="test_node", fn=my_func, deps=["dep1", "dep2"])

        assert node.deps == frozenset({"dep1", "dep2"})

    def test_creates_node_spec_with_input_mapping(self) -> None:
        """FunctionNode creates NodeSpec with input mapping in params."""
        factory = FunctionNode()

        def my_func(input_data: dict[str, Any]) -> dict[str, Any]:
            return input_data

        mapping = {"field1": "$input.source1", "field2": "dep1.output"}
        node = factory(name="test_node", fn=my_func, input_mapping=mapping)

        assert node.params["input_mapping"] == mapping

    def test_resolves_function_from_module_path(self) -> None:
        """FunctionNode resolves function from module path string."""
        factory = FunctionNode()

        # Use a stdlib function
        node = factory(name="json_loads", fn="json.loads")

        assert node.name == "json_loads"
        assert node.fn is not None

    def test_raises_on_invalid_module_path(self) -> None:
        """FunctionNode raises ValueError for invalid module path."""
        factory = FunctionNode()

        with pytest.raises(ValueError, match="Could not import module"):
            factory(name="invalid", fn="nonexistent.module.func")


class TestFunctionNodeUnpackInput:
    """Tests for the unpack_input feature."""

    def test_unpack_input_stored_in_params(self) -> None:
        """unpack_input=True is stored in node params."""
        factory = FunctionNode()

        def my_func(field1: str, field2: int) -> dict[str, Any]:
            return {"field1": field1, "field2": field2}

        node = factory(
            name="test_node",
            fn=my_func,
            input_mapping={"field1": "$input.src1", "field2": "$input.src2"},
            unpack_input=True,
        )

        assert node.params.get("unpack_input") is True

    def test_unpack_input_not_in_params_when_false(self) -> None:
        """unpack_input is not in params when False (default)."""
        factory = FunctionNode()

        def my_func(input_data: dict[str, Any]) -> dict[str, Any]:
            return input_data

        node = factory(name="test_node", fn=my_func)

        assert "unpack_input" not in node.params

    def test_unpack_input_allows_kwargs_only_function(self) -> None:
        """unpack_input=True allows functions with only **kwargs."""
        factory = FunctionNode()

        def my_func(**kwargs: Any) -> dict[str, Any]:
            return kwargs

        # This would fail without unpack_input because first param can't be **kwargs
        node = factory(
            name="test_node",
            fn=my_func,
            input_mapping={"field1": "$input.src"},
            unpack_input=True,
        )

        assert node.name == "test_node"

    def test_unpack_input_allows_no_params_function(self) -> None:
        """unpack_input=True allows functions with no parameters (via **kwargs)."""
        factory = FunctionNode()

        def my_func(**kwargs: Any) -> str:
            return "ok"

        node = factory(name="test_node", fn=my_func, unpack_input=True)

        assert node.name == "test_node"


class TestFunctionNodeUnpackInputExecution:
    """Tests for unpack_input behavior during execution."""

    @pytest.mark.asyncio
    async def test_unpack_input_passes_dict_fields_as_kwargs(self) -> None:
        """With unpack_input=True, dict fields are passed as individual kwargs."""
        factory = FunctionNode()
        received_args: dict[str, Any] = {}

        def capture_func(load_id: str, rate: float) -> dict[str, Any]:
            received_args["load_id"] = load_id
            received_args["rate"] = rate
            return {"status": "ok"}

        node = factory(
            name="test_node",
            fn=capture_func,
            input_mapping={"load_id": "$input.load_id", "rate": "$input.rate"},
            unpack_input=True,
        )

        # Execute the wrapped function
        input_data = {"load_id": "LOAD123", "rate": 1.5}
        result = await node.fn(input_data)

        assert received_args["load_id"] == "LOAD123"
        assert received_args["rate"] == 1.5
        assert result == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_unpack_input_passes_pydantic_fields_as_kwargs(self) -> None:
        """With unpack_input=True, Pydantic model fields are passed as individual kwargs."""
        factory = FunctionNode()
        received_args: dict[str, Any] = {}

        class InputModel(BaseModel):
            load_id: str
            rate: float

        def capture_func(load_id: str, rate: float) -> dict[str, Any]:
            received_args["load_id"] = load_id
            received_args["rate"] = rate
            return {"status": "ok"}

        node = factory(
            name="test_node",
            fn=capture_func,
            input_schema=InputModel,
            unpack_input=True,
        )

        # Execute with Pydantic model
        input_data = InputModel(load_id="LOAD456", rate=2.5)
        result = await node.fn(input_data)

        assert received_args["load_id"] == "LOAD456"
        assert received_args["rate"] == 2.5
        assert result == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_unpack_input_merges_with_ports(self) -> None:
        """With unpack_input=True, ports are merged with unpacked input fields."""
        factory = FunctionNode()
        received_args: dict[str, Any] = {}

        def capture_func(load_id: str, rate: float, db: Any = None) -> dict[str, Any]:
            received_args["load_id"] = load_id
            received_args["rate"] = rate
            received_args["db"] = db
            return {"status": "ok"}

        node = factory(
            name="test_node",
            fn=capture_func,
            input_mapping={"load_id": "$input.load_id", "rate": "$input.rate"},
            unpack_input=True,
        )

        # Execute with ports
        input_data = {"load_id": "LOAD789", "rate": 3.5}
        mock_db = MagicMock()
        await node.fn(input_data, db=mock_db)

        assert received_args["load_id"] == "LOAD789"
        assert received_args["rate"] == 3.5
        assert received_args["db"] is mock_db

    @pytest.mark.asyncio
    async def test_unpack_input_with_async_function(self) -> None:
        """unpack_input works with async functions."""
        factory = FunctionNode()
        received_args: dict[str, Any] = {}

        async def async_capture(load_id: str, rate: float) -> dict[str, Any]:
            await asyncio.sleep(0)  # Simulate async operation
            received_args["load_id"] = load_id
            received_args["rate"] = rate
            return {"status": "async_ok"}

        node = factory(
            name="test_node",
            fn=async_capture,
            input_mapping={"load_id": "$input.load_id", "rate": "$input.rate"},
            unpack_input=True,
        )

        input_data = {"load_id": "ASYNC123", "rate": 4.5}
        result = await node.fn(input_data)

        assert received_args["load_id"] == "ASYNC123"
        assert received_args["rate"] == 4.5
        assert result == {"status": "async_ok"}

    @pytest.mark.asyncio
    async def test_standard_mode_passes_input_data_as_first_arg(self) -> None:
        """Without unpack_input, input_data is passed as first positional argument."""
        factory = FunctionNode()
        received_input: list[Any] = []

        def capture_func(input_data: dict[str, Any]) -> dict[str, Any]:
            received_input.append(input_data)
            return {"status": "ok"}

        node = factory(name="test_node", fn=capture_func)

        input_data = {"field1": "value1", "field2": "value2"}
        await node.fn(input_data)

        assert len(received_input) == 1
        assert received_input[0] == input_data

    @pytest.mark.asyncio
    async def test_unpack_input_with_kwargs_function(self) -> None:
        """unpack_input works with **kwargs functions."""
        factory = FunctionNode()
        received_kwargs: dict[str, Any] = {}

        def kwargs_func(**kwargs: Any) -> dict[str, Any]:
            received_kwargs.update(kwargs)
            return {"status": "ok"}

        node = factory(
            name="test_node",
            fn=kwargs_func,
            input_mapping={"field1": "$input.f1", "field2": "$input.f2"},
            unpack_input=True,
        )

        input_data = {"field1": "val1", "field2": "val2"}
        await node.fn(input_data)

        assert received_kwargs["field1"] == "val1"
        assert received_kwargs["field2"] == "val2"


class TestFunctionNodeValidation:
    """Tests for function validation."""

    def test_validates_function_has_parameters(self) -> None:
        """Standard mode requires function to have at least one parameter."""
        factory = FunctionNode()

        def no_params() -> str:
            return "ok"

        with pytest.raises(ValueError, match="must have at least one parameter"):
            factory(name="test_node", fn=no_params)

    def test_validates_first_param_not_kwargs(self) -> None:
        """Standard mode requires first parameter to not be **kwargs."""
        factory = FunctionNode()

        def kwargs_first(**kwargs: Any) -> str:
            return "ok"

        with pytest.raises(ValueError, match="First parameter cannot be"):
            factory(name="test_node", fn=kwargs_first)

    def test_unpack_input_skips_validation(self) -> None:
        """unpack_input=True skips standard validation."""
        factory = FunctionNode()

        # Function with only **kwargs - normally invalid
        def kwargs_only(**kwargs: Any) -> str:
            return "ok"

        # Should not raise with unpack_input=True
        node = factory(name="test_node", fn=kwargs_only, unpack_input=True)
        assert node.name == "test_node"


class TestFunctionNodeSchemaInference:
    """Tests for schema inference from function annotations."""

    def test_infers_input_schema_from_pydantic_annotation(self) -> None:
        """Infers input schema from Pydantic model type hint."""
        factory = FunctionNode()

        class MyInput(BaseModel):
            field1: str
            field2: int

        def my_func(data: MyInput) -> dict[str, Any]:
            return {}

        node = factory(name="test_node", fn=my_func)

        assert node.in_model is MyInput

    def test_infers_output_schema_from_return_annotation(self) -> None:
        """Infers output schema from Pydantic model return type hint."""
        factory = FunctionNode()

        class MyOutput(BaseModel):
            result: str

        def my_func(data: dict[str, Any]) -> MyOutput:
            return MyOutput(result="ok")

        node = factory(name="test_node", fn=my_func)

        assert node.out_model is MyOutput

    def test_explicit_schema_overrides_inference(self) -> None:
        """Explicit input_schema overrides inferred schema."""
        factory = FunctionNode()

        class InferredInput(BaseModel):
            inferred: str

        class ExplicitInput(BaseModel):
            explicit: str

        def my_func(data: InferredInput) -> dict[str, Any]:
            return {}

        node = factory(name="test_node", fn=my_func, input_schema=ExplicitInput)

        assert node.in_model is ExplicitInput


class TestFunctionNodeHelpers:
    """Tests for helper methods."""

    def test_create_passthrough_mapping(self) -> None:
        """create_passthrough_mapping creates identity mapping."""
        mapping = FunctionNode.create_passthrough_mapping(["field1", "field2"])

        assert mapping == {"field1": "field1", "field2": "field2"}

    def test_create_rename_mapping(self) -> None:
        """create_rename_mapping returns mapping as-is."""
        original = {"new_name": "old_name"}
        mapping = FunctionNode.create_rename_mapping(original)

        assert mapping == original

    def test_create_prefixed_mapping(self) -> None:
        """create_prefixed_mapping creates prefixed source paths."""
        mapping = FunctionNode.create_prefixed_mapping(
            fields=["id", "name"],
            source_node="user_data",
            prefix="user_",
        )

        assert mapping == {
            "user_id": "user_data.id",
            "user_name": "user_data.name",
        }
