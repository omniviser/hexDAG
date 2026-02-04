"""Tests for the expression parser module."""

import pytest

from hexdag.core.expression_parser import ExpressionError, compile_expression


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

    def test_function_call_blocked(self) -> None:
        """Test that function calls are blocked."""
        with pytest.raises(ExpressionError) as exc_info:
            compile_expression("len(items) > 0")
        # Error message indicates Call type is disallowed
        assert "Disallowed expression type: Call" in str(exc_info.value)

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

    def test_conditional_node_expression(self) -> None:
        """Test expression like in ConditionalNode."""
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
        """Test expression like in LoopNode."""
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
