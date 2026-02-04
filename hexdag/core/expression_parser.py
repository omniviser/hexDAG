"""Safe expression parser for YAML conditional expressions.

This module provides a secure way to compile string expressions like
`"node.action == 'ACCEPT'"` into callable predicates without using eval().

Uses Python's AST module with a strict whitelist approach:
- Only allows comparison operators: ==, !=, <, >, <=, >=
- Only allows boolean operators: and, or, not
- Only allows membership: in, not in
- Only allows attribute access and subscript for data extraction
- NO function calls, imports, or arbitrary code execution

Examples
--------
Basic usage::

    from hexdag.core.expression_parser import compile_expression

    # Compile expression to predicate
    pred = compile_expression("action == 'ACCEPT'")
    result = pred({"action": "ACCEPT"}, {})  # True

    # Nested attribute access
    pred = compile_expression("node.response.status == 'success'")
    result = pred({"node": {"response": {"status": "success"}}}, {})  # True

    # Boolean operators
    pred = compile_expression("count > 5 and active == True")
    result = pred({"count": 10, "active": True}, {})  # True

    # State access
    pred = compile_expression("state.iteration < 10")
    result = pred({}, {"iteration": 5})  # True

    # Membership test
    pred = compile_expression("status in ['pending', 'active']")
    result = pred({"status": "active"}, {})  # True
"""

import ast
import operator
from collections.abc import Callable
from typing import Any

from hexdag.core.logging import get_logger

__all__ = ["compile_expression", "ExpressionError"]

logger = get_logger(__name__)


class ExpressionError(Exception):
    """Raised when expression parsing or evaluation fails."""

    def __init__(self, expression: str, reason: str) -> None:
        self.expression = expression
        self.reason = reason
        super().__init__(f"Expression error in '{expression}': {reason}")


# Allowed comparison operators
_COMPARE_OPS: dict[type[ast.cmpop], Callable[[Any, Any], bool]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda x, y: x in y,
    ast.NotIn: lambda x, y: x not in y,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}

# Allowed boolean operators
_BOOL_OPS: dict[type[ast.boolop], Callable[..., bool]] = {
    ast.And: lambda *args: all(args),
    ast.Or: lambda *args: any(args),
}

# Allowed unary operators (return Any since they can be bool or numeric)
_UNARY_OPS: dict[type[ast.unaryop], Callable[..., Any]] = {
    ast.Not: operator.not_,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _validate_ast(node: ast.AST, expression: str) -> None:
    """Validate that an AST node only contains allowed operations.

    Parameters
    ----------
    node : ast.AST
        The AST node to validate
    expression : str
        Original expression for error messages

    Raises
    ------
    ExpressionError
        If the AST contains disallowed operations
    """
    allowed_types = (
        ast.Expression,
        ast.Compare,
        ast.BoolOp,
        ast.UnaryOp,
        ast.BinOp,
        ast.Attribute,
        ast.Subscript,
        ast.Name,
        ast.Constant,
        ast.Load,
        ast.Index,  # Python 3.8 compatibility
        ast.Slice,
        ast.Tuple,
        ast.List,
        ast.Dict,
        # Comparison operators
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.In,
        ast.NotIn,
        ast.Is,
        ast.IsNot,
        # Boolean operators
        ast.And,
        ast.Or,
        ast.Not,
        # Unary operators
        ast.USub,
        ast.UAdd,
        # Binary operators (for arithmetic if needed)
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
    )

    # Check this node
    if not isinstance(node, allowed_types):
        raise ExpressionError(expression, f"Disallowed expression type: {type(node).__name__}")

    # Check for function calls explicitly
    if isinstance(node, ast.Call):
        raise ExpressionError(expression, "Function calls are not allowed in expressions")

    # Recursively check all child nodes
    for child in ast.iter_child_nodes(node):
        _validate_ast(child, expression)


def _get_value(data: dict[str, Any], state: dict[str, Any], path: list[str]) -> Any:
    """Extract value from data or state using a path.

    Parameters
    ----------
    data : dict
        Primary data dict (node outputs)
    state : dict
        Secondary state dict (loop state, etc.)
    path : list[str]
        Path components like ["node", "action"]

    Returns
    -------
    Any
        Extracted value or None if not found
    """
    if not path:
        return None

    # Check if first component refers to "state"
    if path[0] == "state":
        current: Any = state
        path = path[1:]
    else:
        current = data

    for key in path:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(key)
        elif hasattr(current, key):
            current = getattr(current, key)
        else:
            return None

    return current


def _evaluate_node(node: ast.AST, data: dict[str, Any], state: dict[str, Any]) -> Any:
    """Evaluate an AST node against data and state.

    Parameters
    ----------
    node : ast.AST
        AST node to evaluate
    data : dict
        Data dict for variable resolution
    state : dict
        State dict for state variable resolution

    Returns
    -------
    Any
        Result of evaluation
    """
    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Name):
        # Simple variable access
        if node.id == "True":
            return True
        if node.id == "False":
            return False
        if node.id == "None":
            return None
        if node.id == "state":
            return state
        return data.get(node.id)

    if isinstance(node, ast.Attribute):
        # Build path for attribute access
        path = _collect_attribute_path(node)
        return _get_value(data, state, path)

    if isinstance(node, ast.Subscript):
        # Handle subscript access: data["key"] or data[0]
        value = _evaluate_node(node.value, data, state)
        # Handle slice (Python 3.9+ changed ast.Index)
        if isinstance(node.slice, ast.Index):  # Python 3.8
            key = _evaluate_node(node.slice.value, data, state)  # type: ignore[attr-defined]
        else:
            key = _evaluate_node(node.slice, data, state)
        if value is None:
            return None
        if isinstance(value, dict):
            return value.get(key)
        if isinstance(value, (list, tuple)) and isinstance(key, int):
            return value[key] if 0 <= key < len(value) else None
        return None

    if isinstance(node, ast.Compare):
        # Handle chained comparisons: a < b < c
        left = _evaluate_node(node.left, data, state)
        result = True
        for op, comparator in zip(node.ops, node.comparators, strict=False):
            right = _evaluate_node(comparator, data, state)
            op_func = _COMPARE_OPS.get(type(op))
            if op_func is None:
                raise ExpressionError("", f"Unsupported comparison: {type(op).__name__}")
            try:
                result = result and op_func(left, right)
            except TypeError:
                # Handle None comparisons gracefully
                return False
            left = right
        return result

    if isinstance(node, ast.BoolOp):
        # Handle and/or with short-circuit evaluation
        values = [_evaluate_node(v, data, state) for v in node.values]
        op_func = _BOOL_OPS.get(type(node.op))
        if op_func is None:
            raise ExpressionError("", f"Unsupported boolean op: {type(node.op).__name__}")
        return op_func(*values)

    if isinstance(node, ast.UnaryOp):
        operand = _evaluate_node(node.operand, data, state)
        unary_op_func = _UNARY_OPS.get(type(node.op))
        if unary_op_func is None:
            raise ExpressionError("", f"Unsupported unary op: {type(node.op).__name__}")
        return unary_op_func(operand)

    if isinstance(node, ast.List):
        return [_evaluate_node(elt, data, state) for elt in node.elts]

    if isinstance(node, ast.Tuple):
        return tuple(_evaluate_node(elt, data, state) for elt in node.elts)

    if isinstance(node, ast.Dict):
        keys = [_evaluate_node(k, data, state) if k else None for k in node.keys]
        values = [_evaluate_node(v, data, state) for v in node.values]
        return dict(zip(keys, values, strict=False))

    if isinstance(node, ast.BinOp):
        # Handle arithmetic operators
        left = _evaluate_node(node.left, data, state)
        right = _evaluate_node(node.right, data, state)
        bin_ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Mod: operator.mod,
        }
        op_func = bin_ops.get(type(node.op))
        if op_func is None:
            raise ExpressionError("", f"Unsupported binary op: {type(node.op).__name__}")
        return op_func(left, right)

    raise ExpressionError("", f"Unsupported AST node: {type(node).__name__}")


def _collect_attribute_path(node: ast.Attribute) -> list[str]:
    """Collect attribute path from nested attribute access.

    For `node.response.action`, returns ["node", "response", "action"]
    """
    path: list[str] = []
    current: ast.AST = node

    while isinstance(current, ast.Attribute):
        path.append(current.attr)
        current = current.value

    if isinstance(current, ast.Name):
        path.append(current.id)

    return list(reversed(path))


def compile_expression(expression: str) -> Callable[[dict[str, Any], dict[str, Any]], bool]:
    """Compile a string expression into a safe predicate function.

    The compiled predicate takes two arguments:
    - data: dict containing node outputs and other data
    - state: dict containing loop state or other state variables

    Parameters
    ----------
    expression : str
        Expression string like "action == 'ACCEPT'" or "count > 5 and active"

    Returns
    -------
    Callable[[dict, dict], bool]
        Predicate function that returns True/False

    Raises
    ------
    ExpressionError
        If expression is invalid or contains disallowed operations

    Examples
    --------
    >>> pred = compile_expression("action == 'ACCEPT'")
    >>> pred({"action": "ACCEPT"}, {})
    True

    >>> pred = compile_expression("node.status in ['active', 'pending']")
    >>> pred({"node": {"status": "active"}}, {})
    True

    >>> pred = compile_expression("state.iteration < 10")
    >>> pred({}, {"iteration": 5})
    True
    """
    if not expression or not expression.strip():
        raise ExpressionError(expression, "Expression cannot be empty")

    expression = expression.strip()

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ExpressionError(expression, f"Syntax error: {e.msg}") from e

    # Validate AST is safe
    _validate_ast(tree, expression)

    def predicate(data: dict[str, Any], state: dict[str, Any]) -> bool:
        """Evaluate the compiled expression."""
        try:
            result = _evaluate_node(tree.body, data, state)
            return bool(result)
        except Exception as e:
            logger.warning(f"Expression '{expression}' evaluation failed: {e}")
            return False

    return predicate
