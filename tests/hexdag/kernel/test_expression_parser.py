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

    def test_generator_expression_blocked(self) -> None:
        """Test that generator expression is blocked (lazy evaluation)."""
        with pytest.raises(ExpressionError):
            compile_expression("sum(x for x in range(10))")


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


class TestComprehensions:
    """Test list/set/dict comprehension support."""

    def test_list_comp_filter_none(self) -> None:
        """Test filtering None values — the original use case."""
        result = evaluate_expression(
            "[loc for loc in [extracted_origin, extracted_destination] if loc]",
            {"extracted_origin": "NYC", "extracted_destination": None},
        )
        assert result == ["NYC"]

    def test_list_comp_filter_none_all_present(self) -> None:
        """Test when all values are present."""
        result = evaluate_expression(
            "[loc for loc in [extracted_origin, extracted_destination] if loc]",
            {"extracted_origin": "NYC", "extracted_destination": "LAX"},
        )
        assert result == ["NYC", "LAX"]

    def test_list_comp_with_condition(self) -> None:
        """Test list comprehension with comparison filter."""
        result = evaluate_expression(
            "[x for x in items if x > 3]",
            {"items": [1, 2, 3, 4, 5]},
        )
        assert result == [4, 5]

    def test_list_comp_with_transform(self) -> None:
        """Test list comprehension with function call on element."""
        result = evaluate_expression(
            "[upper(name) for name in names]",
            {"names": ["alice", "bob"]},
        )
        assert result == ["ALICE", "BOB"]

    def test_list_comp_with_range(self) -> None:
        """Test list comprehension with range."""
        result = evaluate_expression("[x * 2 for x in range(4)]", {})
        assert result == [0, 2, 4, 6]

    def test_set_comp(self) -> None:
        """Test set comprehension deduplicates."""
        result = evaluate_expression("{x for x in items}", {"items": [1, 2, 2, 3, 3]})
        assert result == {1, 2, 3}

    def test_dict_comp(self) -> None:
        """Test dict comprehension."""
        result = evaluate_expression(
            "{k: v for k, v in zip(keys, vals)}",
            {"keys": ["a", "b"], "vals": [1, 2]},
        )
        assert result == {"a": 1, "b": 2}

    def test_tuple_unpacking_in_comp(self) -> None:
        """Test tuple unpacking in comprehension target."""
        result = evaluate_expression(
            "[k for k, v in zip(names, scores) if v > 80]",
            {"names": ["alice", "bob", "carol"], "scores": [90, 70, 85]},
        )
        assert result == ["alice", "carol"]

    def test_nested_comprehension(self) -> None:
        """Test nested for-clauses in comprehension."""
        result = evaluate_expression(
            "[x for sublist in items for x in sublist]",
            {"items": [[1, 2], [3, 4]]},
        )
        assert result == [1, 2, 3, 4]

    def test_comprehension_still_validates_children(self) -> None:
        """Test that disallowed functions inside comprehensions are blocked."""
        with pytest.raises(ExpressionError):
            compile_expression("[dangerous(x) for x in items]")

    def test_generator_expression_still_blocked(self) -> None:
        """Test that generator expressions remain blocked."""
        with pytest.raises(ExpressionError):
            compile_expression("list(x for x in items)")


class TestMethodCalls:
    """Test whitelisted method calls on objects."""

    def test_dict_get_with_key(self) -> None:
        result = evaluate_expression("data.get('name')", {"data": {"name": "Alice"}})
        assert result == "Alice"

    def test_dict_get_with_default(self) -> None:
        result = evaluate_expression("data.get('missing', 'fallback')", {"data": {}})
        assert result == "fallback"

    def test_dict_keys(self) -> None:
        result = evaluate_expression("list(data.keys())", {"data": {"a": 1, "b": 2}})
        assert sorted(result) == ["a", "b"]

    def test_dict_values(self) -> None:
        result = evaluate_expression("list(data.values())", {"data": {"a": 1, "b": 2}})
        assert sorted(result) == [1, 2]

    def test_str_upper_method(self) -> None:
        result = evaluate_expression("name.upper()", {"name": "hello"})
        assert result == "HELLO"

    def test_str_split_method(self) -> None:
        result = evaluate_expression("text.split(',')", {"text": "a,b,c"})
        assert result == ["a", "b", "c"]

    def test_str_join_method(self) -> None:
        result = evaluate_expression("','.join(parts)", {"parts": ["a", "b", "c"]})
        assert result == "a,b,c"

    def test_str_replace_method(self) -> None:
        result = evaluate_expression("text.replace('old', 'new')", {"text": "old value"})
        assert result == "new value"

    def test_str_startswith_method(self) -> None:
        result = evaluate_expression("name.startswith('he')", {"name": "hello"})
        assert result is True

    def test_method_on_none_returns_none(self) -> None:
        result = evaluate_expression("data.get('key')", {"data": None})
        assert result is None

    def test_disallowed_method_blocked(self) -> None:
        with pytest.raises(ExpressionError):
            compile_expression("obj.__class__()")

    def test_disallowed_method_name(self) -> None:
        with pytest.raises(ExpressionError):
            compile_expression("obj.eval()")

    def test_chained_method_and_function(self) -> None:
        """Method calls can be combined with allowed functions."""
        result = evaluate_expression("len(text.split(','))", {"text": "a,b,c"})
        assert result == 3


class TestShortCircuit:
    """Test short-circuit evaluation of and/or operators."""

    def test_and_short_circuits_on_none(self) -> None:
        """x is not None and x + 1 must not crash when x is None."""
        result = evaluate_expression("x is not None and x + 1", {"x": None})
        assert result is False

    def test_and_evaluates_when_truthy(self) -> None:
        result = evaluate_expression("x is not None and x + 1", {"x": 5})
        assert result == 6

    def test_or_short_circuits_on_truthy(self) -> None:
        result = evaluate_expression("x or 'default'", {"x": "hello"})
        assert result == "hello"

    def test_or_falls_through_to_default(self) -> None:
        result = evaluate_expression("x or 'default'", {"x": ""})
        assert result == "default"

    def test_and_returns_first_falsy_value(self) -> None:
        """Python semantics: 0 and 42 returns 0."""
        result = evaluate_expression("0 and 42", {})
        assert result == 0

    def test_and_returns_last_truthy_value(self) -> None:
        """Python semantics: 'hello' and 42 returns 42."""
        result = evaluate_expression("'hello' and 42", {})
        assert result == 42

    def test_or_returns_first_truthy_value(self) -> None:
        """Python semantics: 'hello' or 42 returns 'hello'."""
        result = evaluate_expression("'hello' or 42", {})
        assert result == "hello"

    def test_or_returns_last_falsy_value(self) -> None:
        """Python semantics: '' or 0 returns 0."""
        result = evaluate_expression("'' or 0", {})
        assert result == 0

    def test_compile_expression_still_returns_bool(self) -> None:
        """compile_expression wraps with bool() — existing behavior preserved."""
        pred = compile_expression("'hello' and 42")
        assert pred({}, {}) is True

    def test_nested_short_circuit(self) -> None:
        """Nested guard: a is not None and b is not None and a + b."""
        result = evaluate_expression(
            "a is not None and b is not None and a + b",
            {"a": 3, "b": 4},
        )
        assert result == 7

        result = evaluate_expression(
            "a is not None and b is not None and a + b",
            {"a": None, "b": 4},
        )
        assert result is False


class TestNegativeIndexing:
    """Test negative list/tuple indexing."""

    def test_negative_one_returns_last(self) -> None:
        result = evaluate_expression("items[-1]", {"items": [1, 2, 3, 4, 5]})
        assert result == 5

    def test_negative_two_returns_second_to_last(self) -> None:
        result = evaluate_expression("items[-2]", {"items": [10, 20, 30]})
        assert result == 20

    def test_out_of_range_negative_returns_none(self) -> None:
        result = evaluate_expression("items[-99]", {"items": [1, 2, 3]})
        assert result is None

    def test_negative_index_on_tuple(self) -> None:
        result = evaluate_expression("t[-1]", {"t": (10, 20, 30)})
        assert result == 30


class TestSlicing:
    """Test slice support in subscript evaluation."""

    def test_basic_slice(self) -> None:
        result = evaluate_expression("items[1:3]", {"items": [0, 1, 2, 3, 4]})
        assert result == [1, 2]

    def test_slice_from_start(self) -> None:
        result = evaluate_expression("items[:2]", {"items": [10, 20, 30, 40]})
        assert result == [10, 20]

    def test_slice_to_end(self) -> None:
        result = evaluate_expression("items[2:]", {"items": [10, 20, 30, 40]})
        assert result == [30, 40]

    def test_negative_slice(self) -> None:
        result = evaluate_expression("items[-2:]", {"items": [10, 20, 30, 40]})
        assert result == [30, 40]

    def test_step_slice(self) -> None:
        result = evaluate_expression("items[::2]", {"items": [0, 1, 2, 3, 4]})
        assert result == [0, 2, 4]

    def test_string_slice(self) -> None:
        result = evaluate_expression("text[:5]", {"text": "hello world"})
        assert result == "hello"

    def test_slice_on_none_returns_none(self) -> None:
        result = evaluate_expression("items[1:3]", {"items": None})
        assert result is None


class TestFloorDivAndPower:
    """Test // (floor division) and ** (power) operators."""

    def test_floor_division(self) -> None:
        result = evaluate_expression("7 // 2", {})
        assert result == 3

    def test_floor_division_negative(self) -> None:
        result = evaluate_expression("-7 // 2", {})
        assert result == -4

    def test_power(self) -> None:
        result = evaluate_expression("2 ** 10", {})
        assert result == 1024

    def test_power_float(self) -> None:
        result = evaluate_expression("9 ** 0.5", {})
        assert result == 3.0

    def test_floor_div_in_expression(self) -> None:
        result = evaluate_expression("total // batch_size", {"total": 100, "batch_size": 7})
        assert result == 14


class TestMutatingMethodsBlocked:
    """Mutating methods (append, extend, update) should be blocked."""

    def test_append_blocked(self) -> None:
        with pytest.raises(ExpressionError):
            compile_expression("items.append(1)")

    def test_extend_blocked(self) -> None:
        with pytest.raises(ExpressionError):
            compile_expression("items.extend([1, 2])")

    def test_update_blocked(self) -> None:
        with pytest.raises(ExpressionError):
            compile_expression("data.update({'key': 'value'})")

    def test_copy_still_allowed(self) -> None:
        result = evaluate_expression("items.copy()", {"items": [1, 2, 3]})
        assert result == [1, 2, 3]


class TestBinopErrorWrapping:
    """Binary operation runtime errors should be wrapped in ExpressionError."""

    def test_division_by_zero(self) -> None:
        with pytest.raises(ExpressionError, match="Division by zero"):
            evaluate_expression("10 / 0", {})

    def test_floor_division_by_zero(self) -> None:
        with pytest.raises(ExpressionError, match="Division by zero"):
            evaluate_expression("10 // 0", {})

    def test_modulo_by_zero(self) -> None:
        with pytest.raises(ExpressionError, match="Division by zero"):
            evaluate_expression("10 % 0", {})

    def test_type_error_in_binop(self) -> None:
        with pytest.raises(ExpressionError, match="Type error"):
            evaluate_expression("'hello' - 5", {})


class TestMissingSentinelBehavior:
    """Tests for MISSING sentinel behavior in path resolution.

    The expression parser returns MISSING for unknown root names but
    None for missing deep fields. These tests document and verify
    that distinction.
    """

    def test_unknown_root_raises_error(self) -> None:
        """Referencing an unknown top-level name raises ExpressionError."""
        with pytest.raises(ExpressionError, match="missing reference"):
            evaluate_expression("nonexistent_var", {"known": 42}, {})

    def test_known_root_missing_deep_returns_none(self) -> None:
        """Known root with missing nested field returns None via default()."""
        result = evaluate_expression(
            "default(data.nonexistent, 'fallback')",
            {"data": {"other": "value"}},
            {},
        )
        assert result == "fallback"

    def test_none_value_propagates(self) -> None:
        """A root that exists but is None propagates correctly."""
        result = evaluate_expression("val", {"val": None}, {})
        assert result is None

    def test_none_deep_field_returns_none(self) -> None:
        """When an intermediate value is None, deeper access returns None."""
        result = evaluate_expression(
            "default(data.child.grandchild, 'missing')",
            {"data": {"child": None}},
            {},
        )
        assert result == "missing"


class TestCtxPipelineContext:
    """Test ctx pipeline context access in expressions."""

    def test_ctx_run_id(self) -> None:
        """ctx.run_id resolves from data dict."""
        ctx = {"run_id": "abc-123", "pipeline_name": "test"}
        result = evaluate_expression("ctx.run_id", {"ctx": ctx}, {})
        assert result == "abc-123"

    def test_ctx_pipeline_name(self) -> None:
        """ctx.pipeline_name resolves correctly."""
        ctx = {"run_id": "", "pipeline_name": "order-processing"}
        result = evaluate_expression("ctx.pipeline_name", {"ctx": ctx}, {})
        assert result == "order-processing"

    def test_ctx_node_name(self) -> None:
        """ctx.node_name resolves correctly."""
        ctx = {"node_name": "analyzer", "run_id": ""}
        result = evaluate_expression("ctx.node_name", {"ctx": ctx}, {})
        assert result == "analyzer"

    def test_ctx_services_list(self) -> None:
        """ctx.services returns a list."""
        ctx = {"services": ["pipeline_memory", "entity_state"]}
        result = evaluate_expression("ctx.services", {"ctx": ctx}, {})
        assert result == ["pipeline_memory", "entity_state"]

    def test_ctx_missing_field_returns_none(self) -> None:
        """ctx.nonexistent returns None (deep field miss)."""
        ctx = {"run_id": "abc"}
        result = evaluate_expression("ctx.nonexistent", {"ctx": ctx}, {})
        assert result is None

    def test_ctx_empty_returns_empty_dict(self) -> None:
        """ctx resolves to empty dict when not injected."""
        result = evaluate_expression("ctx", {}, {})
        assert result == {}

    def test_ctx_in_comparison(self) -> None:
        """ctx fields work in comparison expressions."""
        ctx = {"pipeline_name": "order-processing"}
        pred = compile_expression("ctx.pipeline_name == 'order-processing'")
        assert pred({"ctx": ctx}, {}) is True
        assert pred({"ctx": {"pipeline_name": "other"}}, {}) is False

    def test_ctx_in_string_concatenation(self) -> None:
        """ctx fields work in string expressions."""
        ctx = {"run_id": "xyz"}
        result = evaluate_expression("'run-' + ctx.run_id", {"ctx": ctx}, {})
        assert result == "run-xyz"

    def test_ctx_does_not_shadow_node_results(self) -> None:
        """ctx doesn't interfere with regular node result access."""
        data = {
            "ctx": {"run_id": "abc"},
            "analyzer": {"score": 0.9},
        }
        result = evaluate_expression("analyzer.score", data, {})
        assert result == 0.9
