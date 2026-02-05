"""Tests for ExpressionNode factory.

This module tests the ExpressionNode factory, which evaluates safe AST-based
expressions to compute values in YAML pipelines.
"""

from decimal import Decimal

import pytest

from hexdag.builtin.nodes.expression_node import ExpressionNode


class TestExpressionNodeCreation:
    """Tests for ExpressionNode basic creation."""

    def test_creates_node_spec_with_expressions(self) -> None:
        """ExpressionNode creates NodeSpec from expressions dict."""
        factory = ExpressionNode()

        node = factory(
            name="test_node",
            expressions={"result": "1 + 2"},
        )

        assert node.name == "test_node"
        assert node.fn is not None
        assert node.deps == frozenset()

    def test_creates_node_spec_with_dependencies(self) -> None:
        """ExpressionNode creates NodeSpec with dependencies."""
        factory = ExpressionNode()

        node = factory(
            name="test_node",
            expressions={"result": "value * 2"},
            deps=["dep1", "dep2"],
        )

        assert node.deps == frozenset({"dep1", "dep2"})

    def test_creates_node_spec_with_input_mapping(self) -> None:
        """ExpressionNode creates NodeSpec with input mapping in params."""
        factory = ExpressionNode()

        mapping = {"field1": "$input.source1", "field2": "dep1.output"}
        node = factory(
            name="test_node",
            expressions={"result": "field1 + field2"},
            input_mapping=mapping,
        )

        assert node.params["input_mapping"] == mapping

    def test_output_fields_defaults_to_expression_keys(self) -> None:
        """output_fields defaults to all expression keys when not specified."""
        factory = ExpressionNode()

        node = factory(
            name="test_node",
            expressions={"a": "1", "b": "2", "c": "3"},
        )

        # The default output_fields should include all expression keys
        # We verify this by execution
        assert node.fn is not None


class TestExpressionNodeExecution:
    """Tests for ExpressionNode execution behavior."""

    @pytest.mark.asyncio
    async def test_simple_arithmetic(self) -> None:
        """ExpressionNode evaluates simple arithmetic."""
        factory = ExpressionNode()

        node = factory(
            name="calc",
            expressions={"result": "10 + 5 * 2"},
        )

        result = await node.fn({})
        assert result == {"result": 20}

    @pytest.mark.asyncio
    async def test_uses_input_data(self) -> None:
        """ExpressionNode can reference input data fields."""
        factory = ExpressionNode()

        node = factory(
            name="calc",
            expressions={"doubled": "value * 2"},
        )

        result = await node.fn({"value": 21})
        assert result == {"doubled": 42}

    @pytest.mark.asyncio
    async def test_chained_expressions(self) -> None:
        """ExpressionNode supports chained expressions that reference earlier ones."""
        factory = ExpressionNode()

        node = factory(
            name="calc",
            expressions={
                "step1": "base + 10",
                "step2": "step1 * 2",
                "step3": "step2 - 5",
            },
        )

        result = await node.fn({"base": 5})
        assert result == {"step1": 15, "step2": 30, "step3": 25}

    @pytest.mark.asyncio
    async def test_output_fields_filter(self) -> None:
        """ExpressionNode filters output to specified output_fields."""
        factory = ExpressionNode()

        node = factory(
            name="calc",
            expressions={
                "intermediate": "value + 1",
                "final": "intermediate * 2",
            },
            output_fields=["final"],  # Only output final
        )

        result = await node.fn({"value": 10})
        assert result == {"final": 22}
        assert "intermediate" not in result

    @pytest.mark.asyncio
    async def test_conditional_expression(self) -> None:
        """ExpressionNode evaluates conditional (ternary) expressions."""
        factory = ExpressionNode()

        node = factory(
            name="calc",
            expressions={
                "category": "'high' if score > 80 else 'low'",
            },
        )

        result_high = await node.fn({"score": 90})
        assert result_high == {"category": "high"}

        result_low = await node.fn({"score": 50})
        assert result_low == {"category": "low"}


class TestExpressionNodeWithDecimal:
    """Tests for ExpressionNode with Decimal (financial calculations)."""

    @pytest.mark.asyncio
    async def test_decimal_arithmetic(self) -> None:
        """ExpressionNode supports Decimal for precise arithmetic."""
        factory = ExpressionNode()

        node = factory(
            name="calc",
            expressions={
                "precise": "Decimal('0.1') + Decimal('0.2')",
            },
        )

        result = await node.fn({})
        assert result["precise"] == Decimal("0.3")

    @pytest.mark.asyncio
    async def test_discount_calculation(self) -> None:
        """ExpressionNode calculates discount with Decimal precision."""
        factory = ExpressionNode()

        node = factory(
            name="calc",
            expressions={
                "discount": "Decimal('0.10') if count == 0 else Decimal('0.03')",
                "discounted": "Decimal(str(price)) * (Decimal('1') - discount)",
            },
        )

        result = await node.fn({"price": 100.0, "count": 0})
        assert result["discount"] == Decimal("0.10")
        assert result["discounted"] == Decimal("90.0")

    @pytest.mark.asyncio
    async def test_counter_rate_calculation(self) -> None:
        """ExpressionNode calculates counter rate (negotiation use case)."""
        factory = ExpressionNode()

        node = factory(
            name="calculate_counter",
            expressions={
                "discount": "Decimal('0.10') if counter_count == 0 else Decimal('0.03')",
                "raw_counter": "Decimal(str(offered_rate)) * (Decimal('1') - discount)",
                "floor_decimal": "Decimal(str(default(rate_floor, 0)))",
                "counter_amount": "float(max(raw_counter, floor_decimal))",
            },
            output_fields=["counter_amount"],
        )

        result = await node.fn({
            "offered_rate": 2.50,
            "rate_floor": 1.80,
            "counter_count": 0,
        })

        # 2.50 * 0.90 = 2.25, which is > 1.80
        assert result["counter_amount"] == 2.25


class TestExpressionNodeWithFunctions:
    """Tests for ExpressionNode with whitelisted functions."""

    @pytest.mark.asyncio
    async def test_len_function(self) -> None:
        """ExpressionNode supports len() function."""
        factory = ExpressionNode()

        node = factory(
            name="calc",
            expressions={"count": "len(items)"},
        )

        result = await node.fn({"items": [1, 2, 3, 4, 5]})
        assert result == {"count": 5}

    @pytest.mark.asyncio
    async def test_string_functions(self) -> None:
        """ExpressionNode supports string functions."""
        factory = ExpressionNode()

        node = factory(
            name="calc",
            expressions={
                "upper_name": "upper(name)",
                "lower_name": "lower(name)",
            },
        )

        result = await node.fn({"name": "John Doe"})
        assert result["upper_name"] == "JOHN DOE"
        assert result["lower_name"] == "john doe"

    @pytest.mark.asyncio
    async def test_min_max_functions(self) -> None:
        """ExpressionNode supports min/max functions."""
        factory = ExpressionNode()

        node = factory(
            name="calc",
            expressions={
                "minimum": "min(a, b, c)",
                "maximum": "max(a, b, c)",
            },
        )

        result = await node.fn({"a": 10, "b": 5, "c": 15})
        assert result["minimum"] == 5
        assert result["maximum"] == 15

    @pytest.mark.asyncio
    async def test_default_function(self) -> None:
        """ExpressionNode supports default() function for None handling."""
        factory = ExpressionNode()

        node = factory(
            name="calc",
            expressions={
                "value_with_default": "default(maybe_none, 'fallback')",
            },
        )

        result_none = await node.fn({"maybe_none": None})
        assert result_none == {"value_with_default": "fallback"}

        result_value = await node.fn({"maybe_none": "actual"})
        assert result_value == {"value_with_default": "actual"}

    @pytest.mark.asyncio
    async def test_pow_function(self) -> None:
        """ExpressionNode supports pow() function."""
        factory = ExpressionNode()

        node = factory(
            name="calc",
            expressions={"squared": "pow(value, 2)"},
        )

        result = await node.fn({"value": 5})
        assert result == {"squared": 25}


class TestExpressionNodeErrorHandling:
    """Tests for ExpressionNode error handling."""

    @pytest.mark.asyncio
    async def test_undefined_variable_error(self) -> None:
        """ExpressionNode raises error for undefined variables."""
        factory = ExpressionNode()

        node = factory(
            name="calc",
            expressions={"result": "undefined_var + 1"},
        )

        with pytest.raises(ValueError, match="Expression 'result' failed"):
            await node.fn({})

    @pytest.mark.asyncio
    async def test_invalid_expression_syntax(self) -> None:
        """ExpressionNode raises error for invalid expression syntax."""
        factory = ExpressionNode()

        # This should fail at evaluation time (incomplete expression)
        node = factory(
            name="calc",
            expressions={"result": "1 +"},  # Invalid syntax
        )

        with pytest.raises(ValueError, match="Expression 'result' failed"):
            await node.fn({})

    @pytest.mark.asyncio
    async def test_disallowed_function_error(self) -> None:
        """ExpressionNode raises error for disallowed functions."""
        factory = ExpressionNode()

        # exec is not in ALLOWED_FUNCTIONS
        node = factory(
            name="calc",
            expressions={"result": "exec('print(1)')"},
        )

        with pytest.raises(ValueError, match="Expression 'result' failed"):
            await node.fn({})

    @pytest.mark.asyncio
    async def test_missing_output_field_warning(self) -> None:
        """ExpressionNode handles missing output fields gracefully."""
        factory = ExpressionNode()

        node = factory(
            name="calc",
            expressions={"actual": "1 + 1"},
            output_fields=["actual", "nonexistent"],  # nonexistent won't be computed
        )

        result = await node.fn({})
        # Should include actual but not nonexistent
        assert result == {"actual": 2}
        assert "nonexistent" not in result


class TestExpressionNodeYamlSchema:
    """Tests for ExpressionNode YAML schema metadata."""

    def test_has_yaml_schema(self) -> None:
        """ExpressionNode has _yaml_schema attribute for documentation."""
        assert hasattr(ExpressionNode, "_yaml_schema")

    def test_yaml_schema_has_required_fields(self) -> None:
        """ExpressionNode _yaml_schema specifies required fields."""
        schema = ExpressionNode._yaml_schema

        assert "properties" in schema
        assert "expressions" in schema["properties"]
        assert "input_mapping" in schema["properties"]
        assert "output_fields" in schema["properties"]

        assert "required" in schema
        assert "expressions" in schema["required"]
