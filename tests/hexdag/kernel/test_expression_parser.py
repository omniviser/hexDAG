"""Tests for the expression parser module."""

from decimal import Decimal

import pytest

from hexdag.kernel.expression_parser import (
    ALLOWED_FUNCTIONS,
    ExpressionError,
    compile_expression,
    evaluate_expression,
)


class TestBasicComparisons:
    """Test basic comparison operators."""

    def test_equality(self) -> None:
        """Test == operator."""
        pred = compile_expression("action == 'ACCEPT'")
        assert pred({"action": "ACCEPT"}, {}) is True
        assert pred({"action": "REJECT"}, {}) is False

    def test_inequality(self) -> None:
        """Test != operator."""
        pred = compile_expression("status != 'failed'")
        assert pred({"status": "success"}, {}) is True
        assert pred({"status": "failed"}, {}) is False

    def test_less_than(self) -> None:
        """Test < operator."""
        pred = compile_expression("count < 10")
        assert pred({"count": 5}, {}) is True
        assert pred({"count": 10}, {}) is False
        assert pred({"count": 15}, {}) is False

    def test_less_than_or_equal(self) -> None:
        """Test <= operator."""
        pred = compile_expression("count <= 10")
        assert pred({"count": 5}, {}) is True
        assert pred({"count": 10}, {}) is True
        assert pred({"count": 15}, {}) is False

    def test_greater_than(self) -> None:
        """Test > operator."""
        pred = compile_expression("confidence > 0.8")
        assert pred({"confidence": 0.9}, {}) is True
        assert pred({"confidence": 0.8}, {}) is False
        assert pred({"confidence": 0.5}, {}) is False

    def test_greater_than_or_equal(self) -> None:
        """Test >= operator."""
        pred = compile_expression("score >= 50")
        assert pred({"score": 100}, {}) is True
        assert pred({"score": 50}, {}) is True
        assert pred({"score": 49}, {}) is False


class TestBooleanOperators:
    """Test boolean operators."""

    def test_and_operator(self) -> None:
        """Test and operator."""
        pred = compile_expression("active == True and count > 5")
        assert pred({"active": True, "count": 10}, {}) is True
        assert pred({"active": True, "count": 3}, {}) is False
        assert pred({"active": False, "count": 10}, {}) is False

    def test_or_operator(self) -> None:
        """Test or operator."""
        pred = compile_expression("status == 'active' or status == 'pending'")
        assert pred({"status": "active"}, {}) is True
        assert pred({"status": "pending"}, {}) is True
        assert pred({"status": "failed"}, {}) is False

    def test_not_operator(self) -> None:
        """Test not operator."""
        pred = compile_expression("not done")
        assert pred({"done": False}, {}) is True
        assert pred({"done": True}, {}) is False

    def test_complex_boolean(self) -> None:
        """Test complex boolean expressions."""
        pred = compile_expression("(a > 5 and b < 10) or c == True")
        assert pred({"a": 6, "b": 5, "c": False}, {}) is True
        assert pred({"a": 4, "b": 5, "c": True}, {}) is True
        assert pred({"a": 4, "b": 5, "c": False}, {}) is False


class TestMembershipOperators:
    """Test in/not in operators."""

    def test_in_list(self) -> None:
        """Test in with list."""
        pred = compile_expression("status in ['active', 'pending']")
        assert pred({"status": "active"}, {}) is True
        assert pred({"status": "pending"}, {}) is True
        assert pred({"status": "failed"}, {}) is False

    def test_not_in_list(self) -> None:
        """Test not in with list."""
        pred = compile_expression("action not in ['skip', 'ignore']")
        assert pred({"action": "process"}, {}) is True
        assert pred({"action": "skip"}, {}) is False


class TestAttributeAccess:
    """Test nested attribute access."""

    def test_single_level(self) -> None:
        """Test single level attribute access."""
        pred = compile_expression("node.action == 'ACCEPT'")
        assert pred({"node": {"action": "ACCEPT"}}, {}) is True
        assert pred({"node": {"action": "REJECT"}}, {}) is False

    def test_deep_nesting(self) -> None:
        """Test deeply nested attribute access."""
        pred = compile_expression("response.data.status == 'success'")
        assert pred({"response": {"data": {"status": "success"}}}, {}) is True
        assert pred({"response": {"data": {"status": "failed"}}}, {}) is False

    def test_missing_attribute(self) -> None:
        """Test missing attribute returns None/False."""
        pred = compile_expression("node.missing == 'value'")
        assert pred({"node": {}}, {}) is False

    def test_none_in_path(self) -> None:
        """Test None in path returns False."""
        pred = compile_expression("node.nested.value == 'test'")
        assert pred({"node": {"nested": None}}, {}) is False
        assert pred({"node": None}, {}) is False


class TestStateAccess:
    """Test state dictionary access."""

    def test_state_access(self) -> None:
        """Test accessing state dict."""
        pred = compile_expression("state.iteration < 10")
        assert pred({}, {"iteration": 5}) is True
        assert pred({}, {"iteration": 15}) is False

    def test_state_nested(self) -> None:
        """Test nested state access."""
        pred = compile_expression("state.loop.count > 0")
        assert pred({}, {"loop": {"count": 5}}) is True
        assert pred({}, {"loop": {"count": 0}}) is False

    def test_state_and_data(self) -> None:
        """Test combining state and data access."""
        pred = compile_expression("action == 'continue' and state.iteration < 10")
        assert pred({"action": "continue"}, {"iteration": 5}) is True
        assert pred({"action": "continue"}, {"iteration": 15}) is False
        assert pred({"action": "stop"}, {"iteration": 5}) is False


class TestSubscriptAccess:
    """Test subscript (bracket) access."""

    def test_dict_subscript(self) -> None:
        """Test dict subscript access."""
        pred = compile_expression("data['key'] == 'value'")
        assert pred({"data": {"key": "value"}}, {}) is True
        assert pred({"data": {"key": "other"}}, {}) is False

    def test_list_subscript(self) -> None:
        """Test list subscript access."""
        pred = compile_expression("items[0] == 'first'")
        assert pred({"items": ["first", "second"]}, {}) is True
        assert pred({"items": ["other", "second"]}, {}) is False


class TestLiterals:
    """Test literal values."""

    def test_boolean_literals(self) -> None:
        """Test True/False literals."""
        pred = compile_expression("active == True")
        assert pred({"active": True}, {}) is True
        assert pred({"active": False}, {}) is False

        pred = compile_expression("disabled == False")
        assert pred({"disabled": False}, {}) is True

    def test_none_literal(self) -> None:
        """Test None literal."""
        pred = compile_expression("value == None")
        assert pred({"value": None}, {}) is True
        assert pred({"value": "something"}, {}) is False

    def test_numeric_literals(self) -> None:
        """Test numeric literals."""
        pred = compile_expression("count == 42")
        assert pred({"count": 42}, {}) is True

        pred = compile_expression("ratio == 3.14")
        assert pred({"ratio": 3.14}, {}) is True


class TestSecurityValidation:
    """Test security validations."""

    def test_allowed_function_works(self) -> None:
        """Test that whitelisted functions are allowed."""
        # len() is now allowed
        pred = compile_expression("len(items) > 0")
        assert pred({"items": [1, 2, 3]}, {}) is True
        assert pred({"items": []}, {}) is False

    def test_disallowed_function_blocked(self) -> None:
        """Test that non-whitelisted functions are blocked."""
        with pytest.raises(ExpressionError) as exc_info:
            compile_expression("dangerous_func(x)")
        assert "not allowed" in str(exc_info.value)

    def test_import_blocked(self) -> None:
        """Test that import is blocked."""
        with pytest.raises(ExpressionError):
            compile_expression("__import__('os').system('ls')")

    def test_exec_blocked(self) -> None:
        """Test that exec-like constructs are blocked."""
        with pytest.raises(ExpressionError):
            compile_expression("exec('print(1)')")

    def test_eval_blocked(self) -> None:
        """Test that eval is blocked."""
        with pytest.raises(ExpressionError):
            compile_expression("eval('1+1')")

    def test_lambda_blocked(self) -> None:
        """Test that lambda is blocked."""
        with pytest.raises(ExpressionError):
            compile_expression("(lambda: True)()")

    def test_comprehension_blocked(self) -> None:
        """Test that list comprehension is blocked."""
        with pytest.raises(ExpressionError):
            compile_expression("[x for x in range(10)]")


class TestErrorHandling:
    """Test error handling."""

    def test_empty_expression(self) -> None:
        """Test empty expression raises error."""
        with pytest.raises(ExpressionError) as exc_info:
            compile_expression("")
        assert "cannot be empty" in str(exc_info.value)

    def test_whitespace_only(self) -> None:
        """Test whitespace-only expression raises error."""
        with pytest.raises(ExpressionError) as exc_info:
            compile_expression("   ")
        assert "cannot be empty" in str(exc_info.value)

    def test_syntax_error(self) -> None:
        """Test syntax error in expression."""
        with pytest.raises(ExpressionError) as exc_info:
            compile_expression("a == ==")
        assert "Syntax error" in str(exc_info.value)

    def test_missing_key_returns_false(self) -> None:
        """Test that missing keys return False, not error."""
        pred = compile_expression("missing_key == 'value'")
        # Should return False, not raise
        assert pred({}, {}) is False


class TestRealWorldExamples:
    """Test real-world usage examples."""

    def test_conditional_expression(self) -> None:
        """Test conditional branching expression."""
        # Simulating output from a previous node
        node_output = {
            "determine_action": {
                "result": {
                    "action": "ACCEPT",
                    "confidence": 0.95,
                }
            }
        }

        pred = compile_expression("determine_action.result.action == 'ACCEPT'")
        assert pred(node_output, {}) is True

        pred = compile_expression(
            "determine_action.result.action == 'ACCEPT' and "
            "determine_action.result.confidence > 0.9"
        )
        assert pred(node_output, {}) is True

    def test_loop_condition_expression(self) -> None:
        """Test loop condition expression with state."""
        pred = compile_expression("state.iteration < 10 and not state.done")
        assert pred({}, {"iteration": 5, "done": False}) is True
        assert pred({}, {"iteration": 5, "done": True}) is False
        assert pred({}, {"iteration": 15, "done": False}) is False

    def test_status_routing(self) -> None:
        """Test status-based routing expression."""
        pred = compile_expression("status in ['pending', 'processing'] and retry_count < 3")
        assert pred({"status": "pending", "retry_count": 0}, {}) is True
        assert pred({"status": "failed", "retry_count": 0}, {}) is False
        assert pred({"status": "pending", "retry_count": 5}, {}) is False


class TestChainedComparisons:
    """Test chained comparisons."""

    def test_range_check(self) -> None:
        """Test range check with chained comparison."""
        pred = compile_expression("0 <= value <= 100")
        assert pred({"value": 50}, {}) is True
        assert pred({"value": 0}, {}) is True
        assert pred({"value": 100}, {}) is True
        assert pred({"value": -1}, {}) is False
        assert pred({"value": 101}, {}) is False


class TestTernaryConditional:
    """Test ternary conditional expressions (a if condition else b)."""

    def test_simple_ternary(self) -> None:
        """Test simple ternary conditional."""
        pred = compile_expression("'yes' if active else 'no'")
        assert pred({"active": True}, {}) is True  # 'yes' is truthy
        # Use evaluate_expression for actual value
        result = evaluate_expression("'yes' if active else 'no'", {"active": True})
        assert result == "yes"
        result = evaluate_expression("'yes' if active else 'no'", {"active": False})
        assert result == "no"

    def test_ternary_with_comparison(self) -> None:
        """Test ternary with comparison condition."""
        result = evaluate_expression("'high' if score > 80 else 'low'", {"score": 90})
        assert result == "high"
        result = evaluate_expression("'high' if score > 80 else 'low'", {"score": 70})
        assert result == "low"

    def test_ternary_with_numeric_result(self) -> None:
        """Test ternary returning numeric values."""
        result = evaluate_expression("10 if flag else 0", {"flag": True})
        assert result == 10
        result = evaluate_expression("10 if flag else 0", {"flag": False})
        assert result == 0

    def test_nested_ternary(self) -> None:
        """Test nested ternary conditionals."""
        expr = "'high' if score > 80 else ('medium' if score > 50 else 'low')"
        assert evaluate_expression(expr, {"score": 90}) == "high"
        assert evaluate_expression(expr, {"score": 70}) == "medium"
        assert evaluate_expression(expr, {"score": 30}) == "low"

    def test_ternary_with_boolean_operators(self) -> None:
        """Test ternary with complex boolean condition."""
        expr = "'approved' if score > 80 and verified else 'pending'"
        assert evaluate_expression(expr, {"score": 90, "verified": True}) == "approved"
        assert evaluate_expression(expr, {"score": 90, "verified": False}) == "pending"
        assert evaluate_expression(expr, {"score": 70, "verified": True}) == "pending"

    def test_ternary_with_function_calls(self) -> None:
        """Test ternary with function calls in branches."""
        expr = "upper(name) if uppercase else lower(name)"
        assert evaluate_expression(expr, {"name": "John", "uppercase": True}) == "JOHN"
        assert evaluate_expression(expr, {"name": "John", "uppercase": False}) == "john"

    def test_ternary_in_conditional_context(self) -> None:
        """Test ternary expressions for conditional branching."""
        # Common pattern: selecting discount based on customer tier
        expr = "0.2 if tier == 'gold' else (0.1 if tier == 'silver' else 0)"
        assert evaluate_expression(expr, {"tier": "gold"}) == 0.2
        assert evaluate_expression(expr, {"tier": "silver"}) == 0.1
        assert evaluate_expression(expr, {"tier": "bronze"}) == 0


class TestDecimalSupport:
    """Test Decimal type support for financial calculations."""

    def test_decimal_in_allowed_functions(self) -> None:
        """Test that Decimal is in ALLOWED_FUNCTIONS."""
        assert "Decimal" in ALLOWED_FUNCTIONS
        assert ALLOWED_FUNCTIONS["Decimal"] is Decimal

    def test_decimal_creation(self) -> None:
        """Test creating Decimal values."""
        result = evaluate_expression("Decimal('0.1')", {})
        assert result == Decimal("0.1")
        assert isinstance(result, Decimal)

    def test_decimal_arithmetic(self) -> None:
        """Test Decimal arithmetic operations."""
        # Addition
        result = evaluate_expression("Decimal('0.1') + Decimal('0.2')", {})
        assert result == Decimal("0.3")

        # Subtraction
        result = evaluate_expression("Decimal('1.0') - Decimal('0.3')", {})
        assert result == Decimal("0.7")

        # Multiplication
        result = evaluate_expression("Decimal('2.5') * Decimal('0.4')", {})
        assert result == Decimal("1.00")

        # Division
        result = evaluate_expression("Decimal('1.0') / Decimal('3.0')", {})
        assert result == Decimal("1.0") / Decimal("3.0")

    def test_decimal_comparison(self) -> None:
        """Test Decimal comparisons."""
        pred = compile_expression("Decimal('0.1') + Decimal('0.2') == Decimal('0.3')")
        assert pred({}, {}) is True  # Decimal avoids float precision issues

    def test_decimal_with_variables(self) -> None:
        """Test Decimal with variable values."""
        result = evaluate_expression("Decimal(str(price)) * Decimal('0.9')", {"price": 100.0})
        assert result == Decimal("90.0")

    def test_decimal_ternary_discount(self) -> None:
        """Test Decimal in ternary expression for discount calculation."""
        expr = "Decimal('0.10') if count == 0 else Decimal('0.03')"
        assert evaluate_expression(expr, {"count": 0}) == Decimal("0.10")
        assert evaluate_expression(expr, {"count": 1}) == Decimal("0.03")

    def test_decimal_complex_calculation(self) -> None:
        """Test complex Decimal calculation like in negotiation flow."""
        # Calculate: offered_rate * (1 - discount)
        expr = "Decimal(str(offered_rate)) * (Decimal('1') - Decimal('0.1'))"
        result = evaluate_expression(expr, {"offered_rate": 2.50})
        assert result == Decimal("2.250")

    def test_decimal_max_function(self) -> None:
        """Test max function with Decimal values."""
        result = evaluate_expression(
            "max(Decimal('1.5'), Decimal('2.0'), Decimal('1.8'))",
            {},
        )
        assert result == Decimal("2.0")

    def test_decimal_to_float_conversion(self) -> None:
        """Test converting Decimal result to float."""
        result = evaluate_expression("float(Decimal('2.25'))", {})
        assert result == 2.25
        assert isinstance(result, float)


class TestAdditionalFunctions:
    """Test additional allowed functions (pow, format)."""

    def test_pow_function(self) -> None:
        """Test pow function."""
        assert "pow" in ALLOWED_FUNCTIONS
        result = evaluate_expression("pow(2, 3)", {})
        assert result == 8

        result = evaluate_expression("pow(value, 2)", {"value": 5})
        assert result == 25

    def test_pow_with_decimal(self) -> None:
        """Test pow with Decimal (for compound interest etc.)."""
        result = evaluate_expression("pow(Decimal('1.05'), 2)", {})
        # Note: pow with Decimal might return float, that's OK
        assert abs(float(result) - 1.1025) < 0.0001

    def test_format_function(self) -> None:
        """Test format function."""
        assert "format" in ALLOWED_FUNCTIONS
        result = evaluate_expression("format(value, '.2f')", {"value": 3.14159})
        assert result == "3.14"

    def test_format_with_decimal(self) -> None:
        """Test format with Decimal."""
        result = evaluate_expression("format(Decimal('1234.5'), ',.2f')", {})
        assert result == "1,234.50"


class TestConditionalExpressions:
    """Test expressions commonly used in conditional branching."""

    def test_negotiation_rate_comparison(self) -> None:
        """Test rate comparison expression."""
        pred = compile_expression("extract_offer.rate <= get_context.load.target_rate")
        data = {
            "extract_offer": {"rate": 2.0},
            "get_context": {"load": {"target_rate": 2.5}},
        }
        assert pred(data, {}) is True

        data["extract_offer"]["rate"] = 3.0
        assert pred(data, {}) is False

    def test_confidence_threshold(self) -> None:
        """Test confidence threshold expression."""
        pred = compile_expression(
            "extract_offer.confidence < get_context.system_config.confidence_threshold"
        )
        data = {
            "extract_offer": {"confidence": 0.4},
            "get_context": {"system_config": {"confidence_threshold": 0.5}},
        }
        assert pred(data, {}) is True

        data["extract_offer"]["confidence"] = 0.6
        assert pred(data, {}) is False

    def test_status_check(self) -> None:
        """Test status equality check."""
        pred = compile_expression("get_context.negotiation.status != 'ACTIVE'")
        data = {"get_context": {"negotiation": {"status": "CLOSED"}}}
        assert pred(data, {}) is True

        data["get_context"]["negotiation"]["status"] = "ACTIVE"
        assert pred(data, {}) is False

    def test_boolean_flag_check(self) -> None:
        """Test boolean flag check."""
        pred = compile_expression("get_context.load.winner_locked == True")
        data = {"get_context": {"load": {"winner_locked": True}}}
        assert pred(data, {}) is True

        data["get_context"]["load"]["winner_locked"] = False
        assert pred(data, {}) is False

    def test_combined_conditions(self) -> None:
        """Test multiple conditions combined with 'and'."""
        pred = compile_expression(
            "extract_offer.confidence >= 0.8 and "
            "extract_offer.rate <= get_context.target_rate and "
            "get_context.status == 'ACTIVE'"
        )
        data = {
            "extract_offer": {"confidence": 0.9, "rate": 2.0},
            "get_context": {"target_rate": 2.5, "status": "ACTIVE"},
        }
        assert pred(data, {}) is True

        # Fail on confidence
        data["extract_offer"]["confidence"] = 0.7
        assert pred(data, {}) is False

    def test_null_safe_access(self) -> None:
        """Test handling of missing/null values."""
        pred = compile_expression("default(node.value, 0) > 5")
        assert pred({"node": {"value": 10}}, {}) is True
        assert pred({"node": {"value": None}}, {}) is False
        assert pred({"node": {}}, {}) is False

    def test_counter_count_condition(self) -> None:
        """Test counter count for discount selection."""
        # First counter (count == 0) gets higher discount
        expr = "Decimal('0.10') if counter_count == 0 else Decimal('0.03')"
        assert evaluate_expression(expr, {"counter_count": 0}) == Decimal("0.10")
        assert evaluate_expression(expr, {"counter_count": 1}) == Decimal("0.03")
        assert evaluate_expression(expr, {"counter_count": 5}) == Decimal("0.03")


class TestCompileExpressionCaching:
    """Test that compile_expression uses lru_cache correctly."""

    def test_same_expression_returns_same_predicate(self) -> None:
        """Identical expression strings return the exact same callable object."""
        pred1 = compile_expression("status == 'active'")
        pred2 = compile_expression("status == 'active'")
        assert pred1 is pred2

    def test_different_expressions_return_different_predicates(self) -> None:
        pred1 = compile_expression("x > 0")
        pred2 = compile_expression("x < 0")
        assert pred1 is not pred2

    def test_cached_predicate_still_works(self) -> None:
        """Cached predicates produce correct results with different data."""
        pred = compile_expression("count > 5 and active == True")
        assert pred({"count": 10, "active": True}, {}) is True
        assert pred({"count": 3, "active": True}, {}) is False
        # Call again — same object from cache
        pred2 = compile_expression("count > 5 and active == True")
        assert pred2({"count": 10, "active": True}, {}) is True
