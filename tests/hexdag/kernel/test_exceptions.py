"""Tests for the core exceptions module.

This module tests the hexDAG exception hierarchy.
"""

from __future__ import annotations

import pytest

from hexdag.kernel.exceptions import (
    ConfigurationError,
    DependencyError,
    HexDAGError,
    OrchestratorError,
    ParseError,
    ResourceNotFoundError,
    TypeMismatchError,
    ValidationError,
)


class TestHexDAGError:
    """Tests for HexDAGError base exception."""

    def test_basic_creation(self) -> None:
        """Test creating a basic HexDAGError."""
        error = HexDAGError("Something went wrong")
        assert str(error) == "Something went wrong"

    def test_inheritance(self) -> None:
        """Test that HexDAGError inherits from Exception."""
        error = HexDAGError("test")
        assert isinstance(error, Exception)

    def test_can_be_caught_as_exception(self) -> None:
        """Test that HexDAGError can be caught as Exception."""
        with pytest.raises(HexDAGError):
            raise HexDAGError("test")


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_basic_creation(self) -> None:
        """Test creating a ConfigurationError."""
        error = ConfigurationError("pipeline", "YAML file not found")
        assert "pipeline" in str(error)
        assert "YAML file not found" in str(error)
        assert error.component == "pipeline"
        assert error.reason == "YAML file not found"

    def test_inherits_from_hexdag_error(self) -> None:
        """Test that ConfigurationError inherits from HexDAGError."""
        error = ConfigurationError("test", "reason")
        assert isinstance(error, HexDAGError)


class TestValidationError:
    """Tests for ValidationError."""

    def test_without_value(self) -> None:
        """Test creating ValidationError without value."""
        error = ValidationError("max_iterations", "must be positive")
        assert "max_iterations" in str(error)
        assert "must be positive" in str(error)
        assert error.field == "max_iterations"
        assert error.constraint == "must be positive"
        assert error.value is None

    def test_with_value(self) -> None:
        """Test creating ValidationError with value."""
        error = ValidationError("max_iterations", "must be positive", value=-1)
        assert "max_iterations" in str(error)
        assert "must be positive" in str(error)
        assert "-1" in str(error)
        assert error.value == -1

    def test_inherits_from_hexdag_error(self) -> None:
        """Test that ValidationError inherits from HexDAGError."""
        error = ValidationError("field", "constraint")
        assert isinstance(error, HexDAGError)


class TestParseError:
    """Tests for ParseError."""

    def test_basic_creation(self) -> None:
        """Test creating a ParseError."""
        error = ParseError("Failed to parse JSON from LLM output")
        assert "Failed to parse JSON" in str(error)

    def test_inherits_from_hexdag_error(self) -> None:
        """Test that ParseError inherits from HexDAGError."""
        error = ParseError("test")
        assert isinstance(error, HexDAGError)


class TestResourceNotFoundError:
    """Tests for ResourceNotFoundError."""

    def test_without_available(self) -> None:
        """Test creating ResourceNotFoundError without available list."""
        error = ResourceNotFoundError("pipeline", "my_workflow")
        assert "Pipeline" in str(error)
        assert "my_workflow" in str(error)
        assert "not found" in str(error)
        assert error.resource_type == "pipeline"
        assert error.resource_id == "my_workflow"
        assert error.available is None

    def test_with_available_short_list(self) -> None:
        """Test with a short available list."""
        error = ResourceNotFoundError("adapter", "custom", ["mock", "openai"])
        assert "mock" in str(error)
        assert "openai" in str(error)
        assert error.available == ["mock", "openai"]

    def test_with_available_long_list(self) -> None:
        """Test with a long available list (truncated)."""
        available = ["a", "b", "c", "d", "e", "f", "g", "h"]
        error = ResourceNotFoundError("node", "missing", available)
        assert "... and 3 more" in str(error)

    def test_inherits_from_hexdag_error(self) -> None:
        """Test that ResourceNotFoundError inherits from HexDAGError."""
        error = ResourceNotFoundError("type", "id")
        assert isinstance(error, HexDAGError)


class TestDependencyError:
    """Tests for DependencyError."""

    def test_basic_creation(self) -> None:
        """Test creating a DependencyError."""
        error = DependencyError("llm", "LLM port is required for agent nodes")
        assert "llm" in str(error)
        assert "required for agent nodes" in str(error)
        assert error.dependency == "llm"
        assert error.reason == "LLM port is required for agent nodes"

    def test_inherits_from_hexdag_error(self) -> None:
        """Test that DependencyError inherits from HexDAGError."""
        error = DependencyError("dep", "reason")
        assert isinstance(error, HexDAGError)


class TestTypeMismatchError:
    """Tests for TypeMismatchError."""

    def test_with_type_objects(self) -> None:
        """Test with type objects."""
        error = TypeMismatchError("component", str, dict)
        assert "str" in str(error)
        assert "dict" in str(error)
        assert error.field == "component"
        assert error.expected is str
        assert error.actual is dict

    def test_with_string_types(self) -> None:
        """Test with string type descriptions."""
        error = TypeMismatchError("input", "NodeSpec", "string")
        assert "NodeSpec" in str(error)
        assert "string" in str(error)

    def test_with_value(self) -> None:
        """Test with value included."""
        error = TypeMismatchError("count", int, str, value="not_a_number")
        assert "int" in str(error)
        assert "str" in str(error)
        assert "not_a_number" in str(error)
        assert error.value == "not_a_number"

    def test_without_value(self) -> None:
        """Test without value."""
        error = TypeMismatchError("field", int, str)
        assert error.value is None

    def test_inherits_from_hexdag_error(self) -> None:
        """Test that TypeMismatchError inherits from HexDAGError."""
        error = TypeMismatchError("f", str, int)
        assert isinstance(error, HexDAGError)


class TestOrchestratorError:
    """Tests for OrchestratorError."""

    def test_basic_creation(self) -> None:
        """Test creating an OrchestratorError."""
        error = OrchestratorError("Node 'fetch_data' failed: timeout")
        assert "fetch_data" in str(error)
        assert "timeout" in str(error)

    def test_inherits_from_hexdag_error(self) -> None:
        """Test that OrchestratorError inherits from HexDAGError."""
        error = OrchestratorError("test")
        assert isinstance(error, HexDAGError)


class TestExceptionHierarchy:
    """Tests for the exception hierarchy."""

    def test_all_exceptions_inherit_from_hexdag_error(self) -> None:
        """Test that all custom exceptions inherit from HexDAGError."""
        exceptions = [
            ConfigurationError("c", "r"),
            ValidationError("f", "c"),
            ParseError("p"),
            ResourceNotFoundError("t", "i"),
            DependencyError("d", "r"),
            TypeMismatchError("f", str, int),
            OrchestratorError("o"),
        ]
        for exc in exceptions:
            assert isinstance(exc, HexDAGError)

    def test_catch_all_with_hexdag_error(self) -> None:
        """Test catching all hexDAG exceptions with HexDAGError."""
        exceptions_to_raise = [
            ConfigurationError("c", "r"),
            ValidationError("f", "c"),
            ParseError("p"),
            ResourceNotFoundError("t", "i"),
            DependencyError("d", "r"),
            TypeMismatchError("f", str, int),
            OrchestratorError("o"),
        ]
        for exc in exceptions_to_raise:
            with pytest.raises(HexDAGError):
                raise exc
