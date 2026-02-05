"""Tests for ExpressionNode factory.

This module tests the ExpressionNode factory, which evaluates safe AST-based
expressions to compute values in YAML pipelines. Also tests merge strategies
for aggregating outputs from multiple dependency nodes.
"""

import statistics
from decimal import Decimal

import pytest

from hexdag.builtin.nodes.expression_node import (
    ExpressionNode,
    _apply_merge_strategy,
    _extract_field,
)


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


class TestExpressionNodeValidation:
    """Tests for ExpressionNode validation."""

    def test_requires_expressions_or_merge_strategy(self) -> None:
        """ExpressionNode requires either expressions or merge_strategy."""
        factory = ExpressionNode()

        with pytest.raises(ValueError, match="Either 'expressions' or 'merge_strategy'"):
            factory(name="test_node")

    def test_reduce_requires_reducer(self) -> None:
        """ExpressionNode with merge_strategy='reduce' requires reducer."""
        factory = ExpressionNode()

        with pytest.raises(ValueError, match="'reducer' is required"):
            factory(
                name="test_node",
                merge_strategy="reduce",
                deps=["dep1", "dep2"],
            )


class TestMergeStrategyHelpers:
    """Tests for merge strategy helper functions."""

    def test_extract_field_with_dict(self) -> None:
        """_extract_field extracts from dict using dot notation."""
        data = {"result": {"score": 42, "nested": {"value": 10}}}

        assert _extract_field(data, "result.score") == 42
        assert _extract_field(data, "result.nested.value") == 10
        assert _extract_field(data, None) == data

    def test_extract_field_returns_none_for_missing(self) -> None:
        """_extract_field returns None for missing fields."""
        data = {"result": {"score": 42}}

        assert _extract_field(data, "result.missing") is None
        assert _extract_field(data, "nonexistent") is None

    def test_apply_merge_strategy_dict(self) -> None:
        """_apply_merge_strategy with 'dict' returns passthrough."""
        input_data = {"node1": {"score": 10}, "node2": {"score": 20}}
        dep_order = ["node1", "node2"]

        result = _apply_merge_strategy(
            input_data, "dict", field_path=None, reducer=None, dep_order=dep_order
        )
        assert result == input_data

    def test_apply_merge_strategy_dict_with_extract(self) -> None:
        """_apply_merge_strategy with 'dict' and extract_field."""
        input_data = {"node1": {"score": 10}, "node2": {"score": 20}}
        dep_order = ["node1", "node2"]

        result = _apply_merge_strategy(input_data, "dict", "score", None, dep_order)
        assert result == {"node1": 10, "node2": 20}

    def test_apply_merge_strategy_list(self) -> None:
        """_apply_merge_strategy with 'list' returns ordered list."""
        input_data = {"node1": {"score": 10}, "node2": {"score": 20}}
        dep_order = ["node1", "node2"]

        result = _apply_merge_strategy(input_data, "list", "score", None, dep_order)
        assert result == [10, 20]

    def test_apply_merge_strategy_first(self) -> None:
        """_apply_merge_strategy with 'first' returns first non-None."""
        input_data = {"node1": None, "node2": {"score": 20}, "node3": {"score": 30}}
        dep_order = ["node1", "node2", "node3"]

        result = _apply_merge_strategy(input_data, "first", "score", None, dep_order)
        assert result == 20

    def test_apply_merge_strategy_last(self) -> None:
        """_apply_merge_strategy with 'last' returns last non-None."""
        input_data = {"node1": {"score": 10}, "node2": {"score": 20}, "node3": None}
        dep_order = ["node1", "node2", "node3"]

        result = _apply_merge_strategy(input_data, "last", "score", None, dep_order)
        assert result == 20

    def test_apply_merge_strategy_reduce(self) -> None:
        """_apply_merge_strategy with 'reduce' applies reducer function."""
        input_data = {"node1": {"score": 10}, "node2": {"score": 20}, "node3": {"score": 30}}
        dep_order = ["node1", "node2", "node3"]

        result = _apply_merge_strategy(input_data, "reduce", "score", statistics.mean, dep_order)
        assert result == 20.0  # mean of [10, 20, 30]

    def test_apply_merge_strategy_reduce_filters_none(self) -> None:
        """_apply_merge_strategy with 'reduce' filters out None values."""
        input_data = {"node1": {"score": 10}, "node2": None, "node3": {"score": 30}}
        dep_order = ["node1", "node2", "node3"]

        result = _apply_merge_strategy(input_data, "reduce", "score", statistics.mean, dep_order)
        assert result == 20.0  # mean of [10, 30]


class TestExpressionNodeMergeStrategies:
    """Tests for ExpressionNode merge strategy execution."""

    @pytest.mark.asyncio
    async def test_merge_strategy_list(self) -> None:
        """ExpressionNode with merge_strategy='list' collects values."""
        factory = ExpressionNode()

        node = factory(
            name="collect",
            merge_strategy="list",
            extract_field="score",
            deps=["scorer_1", "scorer_2", "scorer_3"],
        )

        # Simulate multi-dependency input (dict with node names as keys)
        input_data = {
            "scorer_1": {"score": 10},
            "scorer_2": {"score": 20},
            "scorer_3": {"score": 30},
        }

        result = await node.fn(input_data)
        assert result == {"result": [10, 20, 30]}

    @pytest.mark.asyncio
    async def test_merge_strategy_reduce_with_mean(self) -> None:
        """ExpressionNode with merge_strategy='reduce' applies reducer."""
        factory = ExpressionNode()

        node = factory(
            name="average",
            merge_strategy="reduce",
            extract_field="score",
            reducer=statistics.mean,
            deps=["scorer_1", "scorer_2", "scorer_3"],
        )

        input_data = {
            "scorer_1": {"score": 10},
            "scorer_2": {"score": 20},
            "scorer_3": {"score": 30},
        }

        result = await node.fn(input_data)
        assert result == {"result": 20.0}

    @pytest.mark.asyncio
    async def test_merge_strategy_reduce_with_module_path(self) -> None:
        """ExpressionNode with merge_strategy='reduce' resolves module path."""
        factory = ExpressionNode()

        node = factory(
            name="average",
            merge_strategy="reduce",
            extract_field="score",
            reducer="statistics.mean",  # Module path string
            deps=["scorer_1", "scorer_2"],
        )

        input_data = {
            "scorer_1": {"score": 10},
            "scorer_2": {"score": 30},
        }

        result = await node.fn(input_data)
        assert result == {"result": 20.0}

    @pytest.mark.asyncio
    async def test_merge_strategy_first(self) -> None:
        """ExpressionNode with merge_strategy='first' returns first non-None."""
        factory = ExpressionNode()

        node = factory(
            name="fallback",
            merge_strategy="first",
            deps=["primary", "fallback", "cache"],
        )

        input_data = {
            "primary": None,
            "fallback": {"data": "from_fallback"},
            "cache": {"data": "from_cache"},
        }

        result = await node.fn(input_data)
        assert result == {"result": {"data": "from_fallback"}}

    @pytest.mark.asyncio
    async def test_merge_strategy_last(self) -> None:
        """ExpressionNode with merge_strategy='last' returns last non-None."""
        factory = ExpressionNode()

        node = factory(
            name="latest",
            merge_strategy="last",
            deps=["v1", "v2", "v3"],
        )

        input_data = {
            "v1": {"version": 1},
            "v2": {"version": 2},
            "v3": None,
        }

        result = await node.fn(input_data)
        assert result == {"result": {"version": 2}}

    @pytest.mark.asyncio
    async def test_merge_strategy_dict(self) -> None:
        """ExpressionNode with merge_strategy='dict' returns passthrough."""
        factory = ExpressionNode()

        node = factory(
            name="collect_all",
            merge_strategy="dict",
            deps=["node1", "node2"],
        )

        input_data = {
            "node1": {"a": 1},
            "node2": {"b": 2},
        }

        result = await node.fn(input_data)
        assert result == {"result": {"node1": {"a": 1}, "node2": {"b": 2}}}

    @pytest.mark.asyncio
    async def test_merge_with_expressions(self) -> None:
        """ExpressionNode can combine merge strategy with expressions."""
        factory = ExpressionNode()

        node = factory(
            name="merge_and_calculate",
            merge_strategy="list",
            extract_field="score",
            expressions={
                "total": "sum(merged)",
                "average": "sum(merged) / len(merged)",
            },
            output_fields=["total", "average"],
            deps=["s1", "s2", "s3"],
        )

        input_data = {
            "s1": {"score": 10},
            "s2": {"score": 20},
            "s3": {"score": 30},
        }

        result = await node.fn(input_data)
        assert result["total"] == 60
        assert result["average"] == 20.0
