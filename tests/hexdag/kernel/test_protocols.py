"""Tests for the protocols module.

This module tests the structural typing protocols for hexDAG.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from hexdag.kernel.exceptions import TypeMismatchError
from hexdag.kernel.protocols import (
    ComponentWithExecute,
    ConfigurablePort,
    DictConvertible,
    HealthCheckable,
    SchemaProvider,
    has_execute_method,
    is_dict_convertible,
    is_schema_type,
    to_dict,
)


class TestComponentWithExecuteProtocol:
    """Tests for ComponentWithExecute protocol."""

    def test_class_with_execute_matches_protocol(self) -> None:
        """Test that class with execute method matches protocol."""

        class MyTool:
            def execute(self, **kwargs: Any) -> Any:
                return "result"

        tool = MyTool()
        assert isinstance(tool, ComponentWithExecute)

    def test_class_without_execute_does_not_match(self) -> None:
        """Test that class without execute method doesn't match."""

        class NotATool:
            def run(self) -> None:
                pass

        obj = NotATool()
        assert not isinstance(obj, ComponentWithExecute)


class TestConfigurablePortProtocol:
    """Tests for ConfigurablePort protocol."""

    def test_class_with_get_config_class_matches(self) -> None:
        """Test that class with get_config_class matches protocol."""

        class MyConfig:
            pass

        class MyAdapter:
            @classmethod
            def get_config_class(cls) -> type[Any]:
                return MyConfig

        assert isinstance(MyAdapter(), ConfigurablePort)


class TestHealthCheckableProtocol:
    """Tests for HealthCheckable protocol."""

    def test_class_with_ahealth_check_matches(self) -> None:
        """Test that class with ahealth_check matches protocol."""

        class MyAdapter:
            async def ahealth_check(self) -> dict[str, Any]:
                return {"status": "healthy"}

        adapter = MyAdapter()
        assert isinstance(adapter, HealthCheckable)


class TestDictConvertibleProtocol:
    """Tests for DictConvertible protocol."""

    def test_pydantic_model_matches(self) -> None:
        """Test that Pydantic model matches protocol."""

        class MyModel(BaseModel):
            field: str

        model = MyModel(field="value")
        assert isinstance(model, DictConvertible)

    def test_class_with_model_dump_matches(self) -> None:
        """Test that class with model_dump matches protocol."""

        class CustomClass:
            def model_dump(self) -> dict[str, Any]:
                return {"key": "value"}

        obj = CustomClass()
        assert isinstance(obj, DictConvertible)


class TestSchemaProviderProtocol:
    """Tests for SchemaProvider protocol."""

    def test_pydantic_model_class_matches(self) -> None:
        """Test that Pydantic model class matches protocol."""

        class MyModel(BaseModel):
            field: str

        # SchemaProvider is for classes, not instances
        assert issubclass(MyModel, SchemaProvider)


class TestHasExecuteMethod:
    """Tests for has_execute_method function."""

    def test_returns_true_for_component_with_execute(self) -> None:
        """Test returns True for object with execute method."""

        class Tool:
            def execute(self, **kwargs: Any) -> Any:
                return "result"

        assert has_execute_method(Tool()) is True

    def test_returns_false_for_object_without_execute(self) -> None:
        """Test returns False for object without execute method."""

        class NotTool:
            pass

        assert has_execute_method(NotTool()) is False


class TestIsDictConvertible:
    """Tests for is_dict_convertible function."""

    def test_returns_true_for_dict(self) -> None:
        """Test returns True for dict."""
        assert is_dict_convertible({"key": "value"}) is True

    def test_returns_true_for_pydantic_model(self) -> None:
        """Test returns True for Pydantic model."""

        class MyModel(BaseModel):
            field: str

        assert is_dict_convertible(MyModel(field="value")) is True

    def test_returns_false_for_string(self) -> None:
        """Test returns False for string."""
        assert is_dict_convertible("not a dict") is False

    def test_returns_false_for_list(self) -> None:
        """Test returns False for list."""
        assert is_dict_convertible([1, 2, 3]) is False


class TestIsSchemaType:
    """Tests for is_schema_type function."""

    def test_returns_true_for_pydantic_model_class(self) -> None:
        """Test returns True for Pydantic model class."""

        class MyModel(BaseModel):
            field: str

        assert is_schema_type(MyModel) is True

    def test_returns_false_for_pydantic_model_instance(self) -> None:
        """Test returns False for Pydantic model instance."""

        class MyModel(BaseModel):
            field: str

        assert is_schema_type(MyModel(field="val")) is False

    def test_returns_false_for_regular_class(self) -> None:
        """Test returns False for regular class without schema."""

        class RegularClass:
            pass

        assert is_schema_type(RegularClass) is False

    def test_returns_false_for_non_type(self) -> None:
        """Test returns False for non-type objects."""
        assert is_schema_type("not a type") is False
        assert is_schema_type(123) is False
        assert is_schema_type(None) is False


class TestToDict:
    """Tests for to_dict function."""

    def test_returns_dict_unchanged(self) -> None:
        """Test that dict is returned unchanged."""
        original = {"key": "value", "nested": {"inner": 1}}
        result = to_dict(original)
        assert result == original
        assert result is original  # Same object

    def test_converts_pydantic_model(self) -> None:
        """Test that Pydantic model is converted to dict."""

        class MyModel(BaseModel):
            field: str
            number: int

        model = MyModel(field="value", number=42)
        result = to_dict(model)
        assert result == {"field": "value", "number": 42}
        assert isinstance(result, dict)

    def test_converts_custom_dict_convertible(self) -> None:
        """Test that custom class with model_dump is converted."""

        class CustomClass:
            def model_dump(self) -> dict[str, Any]:
                return {"custom": "data"}

        obj = CustomClass()
        result = to_dict(obj)
        assert result == {"custom": "data"}

    def test_raises_type_error_for_string(self) -> None:
        """Test that TypeMismatchError is raised for string."""
        with pytest.raises(TypeMismatchError):
            to_dict("not convertible")

    def test_raises_type_error_for_list(self) -> None:
        """Test that TypeMismatchError is raised for list."""
        with pytest.raises(TypeMismatchError):
            to_dict([1, 2, 3])

    def test_raises_type_error_for_int(self) -> None:
        """Test that TypeMismatchError is raised for int."""
        with pytest.raises(TypeMismatchError):
            to_dict(42)
