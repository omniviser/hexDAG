"""Tests for the input normalization utility."""

from pydantic import BaseModel

from hexdag.kernel.utils.input_normalization import normalize_input


class SampleModel(BaseModel):
    name: str
    value: int


class TestNormalizeInput:
    def test_dict_input(self) -> None:
        result = normalize_input({"key": "val"})
        assert result == {"key": "val"}
        assert isinstance(result, dict)  # shallow copy

    def test_pydantic_model(self) -> None:
        model = SampleModel(name="test", value=42)
        result = normalize_input(model)
        assert result == {"name": "test", "value": 42}

    def test_scalar_wrapped(self) -> None:
        assert normalize_input("hello") == {"input": "hello"}
        assert normalize_input(42) == {"input": 42}
        assert normalize_input(None) == {"input": None}
