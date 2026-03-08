"""Tests for DataNode -- static data node factory (deprecated, delegates to ExpressionNode)."""

from __future__ import annotations

import warnings

import pytest

from hexdag.kernel.domain.dag import NodeSpec
from hexdag.stdlib.nodes.data_node import DataNode, _value_to_expression

# ---------------------------------------------------------------------------
# TestValueToExpression
# ---------------------------------------------------------------------------


class TestValueToExpression:
    """Tests for _value_to_expression helper."""

    def test_static_string(self) -> None:
        """Static strings are wrapped in repr quotes."""
        result = _value_to_expression("hello")
        assert result == "'hello'"

    def test_template_string_passthrough(self) -> None:
        """Template strings ({{var}}) pass through unchanged."""
        result = _value_to_expression("Hello {{name}}!")
        assert result == "Hello {{name}}!"

    def test_integer(self) -> None:
        """Integers use repr."""
        assert _value_to_expression(42) == "42"

    def test_float(self) -> None:
        """Floats use repr."""
        assert _value_to_expression(3.14) == "3.14"

    def test_boolean_true(self) -> None:
        """Booleans use repr (True not 1)."""
        assert _value_to_expression(True) == "True"

    def test_boolean_false(self) -> None:
        """Booleans use repr (False not 0)."""
        assert _value_to_expression(False) == "False"

    def test_none(self) -> None:
        """None returns 'None' string."""
        assert _value_to_expression(None) == "None"

    def test_list(self) -> None:
        """Lists use repr."""
        result = _value_to_expression([1, 2, 3])
        assert result == "[1, 2, 3]"

    def test_dict(self) -> None:
        """Dicts use repr."""
        result = _value_to_expression({"a": 1})
        assert "a" in result
        assert "1" in result


# ---------------------------------------------------------------------------
# TestDataNodeCreation
# ---------------------------------------------------------------------------


class TestDataNodeCreation:
    """Tests for DataNode factory __call__."""

    def setup_method(self) -> None:
        self.factory = DataNode()

    def test_creates_node_spec(self) -> None:
        """Factory creates a valid NodeSpec."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            spec = self.factory(
                name="static_data",
                output={"status": "OK", "code": 200},
            )
        assert isinstance(spec, NodeSpec)
        assert spec.name == "static_data"
        assert spec.fn is not None

    def test_emits_deprecation_warning(self) -> None:
        """DataNode emits DeprecationWarning on call."""
        with pytest.warns(DeprecationWarning, match="DataNode is deprecated"):
            self.factory(
                name="deprecated",
                output={"key": "value"},
            )

    def test_includes_dependencies(self) -> None:
        """Dependencies are passed through to NodeSpec."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            spec = self.factory(
                name="after_step",
                output={"result": "done"},
                deps=["upstream"],
            )
        assert "upstream" in spec.deps

    def test_no_deps_default(self) -> None:
        """No deps defaults to empty."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            spec = self.factory(
                name="standalone",
                output={"value": 1},
            )
        assert len(spec.deps) == 0


# ---------------------------------------------------------------------------
# TestDataNodeExecution
# ---------------------------------------------------------------------------


class TestDataNodeExecution:
    """Tests for DataNode execution (delegates to ExpressionNode)."""

    def setup_method(self) -> None:
        self.factory = DataNode()

    @pytest.mark.asyncio
    async def test_returns_static_output(self) -> None:
        """Executing data node returns the configured static output."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            spec = self.factory(
                name="static",
                output={"action": "REJECTED", "reason": "locked"},
            )
        result = await spec.fn({})
        assert result["action"] == "REJECTED"
        assert result["reason"] == "locked"

    @pytest.mark.asyncio
    async def test_returns_numeric_output(self) -> None:
        """Numeric output values are preserved."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            spec = self.factory(
                name="numbers",
                output={"count": 42, "ratio": 3.14},
            )
        result = await spec.fn({})
        assert result["count"] == 42
        assert result["ratio"] == 3.14

    @pytest.mark.asyncio
    async def test_returns_boolean_output(self) -> None:
        """Boolean output values are preserved."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            spec = self.factory(
                name="flags",
                output={"active": True, "deleted": False},
            )
        result = await spec.fn({})
        assert result["active"] is True
        assert result["deleted"] is False

    @pytest.mark.asyncio
    async def test_ignores_input_data(self) -> None:
        """Data node output is independent of input."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            spec = self.factory(
                name="constant",
                output={"value": "always_same"},
            )
        result1 = await spec.fn({"any": "input"})
        result2 = await spec.fn({"different": "input"})
        assert result1["value"] == result2["value"] == "always_same"
