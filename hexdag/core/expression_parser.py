"""Safe expression parser for YAML conditional expressions.

This module provides a secure way to compile string expressions like
`"node.action == 'ACCEPT'"` into callable predicates without using eval().

Uses Python's AST module with a strict whitelist approach:
- Only allows comparison operators: ==, !=, <, >, <=, >=
- Only allows boolean operators: and, or, not
- Only allows membership: in, not in
- Only allows attribute access and subscript for data extraction
- Only allows whitelisted function calls (see ALLOWED_FUNCTIONS)

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

    # Built-in functions
    pred = compile_expression("len(items) > 0")
    result = pred({"items": [1, 2, 3]}, {})  # True

    pred = compile_expression("upper(name) == 'JOHN'")
    result = pred({"name": "john"}, {})  # True
"""

import ast
import operator
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from functools import lru_cache
from typing import Any

from hexdag.core.logging import get_logger

__all__ = ["compile_expression", "evaluate_expression", "ExpressionError", "ALLOWED_FUNCTIONS"]

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

# Safe built-in functions allowed in expressions
# These are carefully selected to avoid side effects and security risks
ALLOWED_FUNCTIONS: dict[str, Callable[..., Any]] = {
    # Date/time functions
    "now": lambda: datetime.now(),
    "utcnow": lambda: datetime.now(UTC),
    "timestamp": lambda dt: dt.timestamp() if isinstance(dt, datetime) else float(dt),
    # Type conversion functions
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    # Math functions
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    # Collection functions
    "len": len,
    "all": all,
    "any": any,
    "sorted": sorted,
    "reversed": lambda x: list(reversed(x)),
    "list": list,
    "set": set,
    "dict": dict,
    "tuple": tuple,
    # String operations (wrapped to handle non-strings gracefully)
    "lower": lambda s: s.lower() if isinstance(s, str) else str(s).lower(),
    "upper": lambda s: s.upper() if isinstance(s, str) else str(s).upper(),
    "strip": lambda s: s.strip() if isinstance(s, str) else str(s).strip(),
    "lstrip": lambda s: s.lstrip() if isinstance(s, str) else str(s).lstrip(),
    "rstrip": lambda s: s.rstrip() if isinstance(s, str) else str(s).rstrip(),
    "split": lambda s, sep=None: s.split(sep) if isinstance(s, str) else [s],
    "join": lambda sep, items: sep.join(str(i) for i in items),
    "replace": lambda s, old, new: s.replace(old, new) if isinstance(s, str) else s,
    "startswith": lambda s, prefix: s.startswith(prefix) if isinstance(s, str) else False,
    "endswith": lambda s, suffix: s.endswith(suffix) if isinstance(s, str) else False,
    "contains": lambda s, sub: sub in s if isinstance(s, str) else False,
    # Conditional/utility functions
    "default": lambda val, default: val if val is not None else default,
    "coalesce": lambda *args: next((a for a in args if a is not None), None),
    "isnone": lambda x: x is None,
    "isempty": lambda x: x is None or x == "" or x == [] or x == {},
    # Financial/precision math functions
    "Decimal": Decimal,
    "pow": pow,
    "format": format,
}


def _get_function_name(func_node: ast.AST) -> str | None:
    """Extract function name from a Call node's func attribute.

    Parameters
    ----------
    func_node : ast.AST
        The func attribute of an ast.Call node

    Returns
    -------
    str | None
        Function name if it's a simple Name node, None otherwise
    """
    if isinstance(func_node, ast.Name):
        return func_node.id
    return None


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
        ast.IfExp,  # Ternary conditional: a if condition else b
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
        ast.Call,  # Now allowed for whitelisted functions
        ast.keyword,  # For keyword arguments in function calls
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

    # Check for function calls - only allow whitelisted functions
    if isinstance(node, ast.Call):
        func_name = _get_function_name(node.func)
        if func_name is None:
            raise ExpressionError(
                expression,
                "Only simple function calls are allowed (e.g., 'len(x)', not 'obj.method()')",
            )
        if func_name not in ALLOWED_FUNCTIONS:
            allowed_list = ", ".join(sorted(ALLOWED_FUNCTIONS.keys()))
            raise ExpressionError(
                expression,
                f"Function '{func_name}' is not allowed. Allowed functions: {allowed_list}",
            )

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

    if isinstance(node, ast.IfExp):
        # Handle ternary conditional: a if condition else b
        condition = _evaluate_node(node.test, data, state)
        if condition:
            return _evaluate_node(node.body, data, state)
        return _evaluate_node(node.orelse, data, state)

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

    if isinstance(node, ast.Call):
        # Handle whitelisted function calls
        func_name = _get_function_name(node.func)
        if func_name is None or func_name not in ALLOWED_FUNCTIONS:
            raise ExpressionError("", f"Unknown or disallowed function: {func_name}")

        func = ALLOWED_FUNCTIONS[func_name]

        # Evaluate arguments
        args = [_evaluate_node(arg, data, state) for arg in node.args]

        # Evaluate keyword arguments
        kwargs = {}
        for kw in node.keywords:
            if kw.arg is not None:
                kwargs[kw.arg] = _evaluate_node(kw.value, data, state)

        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise ExpressionError("", f"Error calling {func_name}: {e}") from e

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


@lru_cache(maxsize=256)
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


def evaluate_expression(
    expression: str,
    data: dict[str, Any],
    state: dict[str, Any] | None = None,
) -> Any:
    """Evaluate an expression and return the actual result value.

    Unlike compile_expression which returns a boolean predicate, this function
    returns the actual computed value of the expression. Use this for input_mapping
    transformations where you need the actual value, not just True/False.

    Parameters
    ----------
    expression : str
        Expression string like "len(items)" or "upper(name)"
    data : dict
        Data dict for variable resolution
    state : dict | None
        Optional state dict for state variable resolution

    Returns
    -------
    Any
        The actual result of evaluating the expression

    Raises
    ------
    ExpressionError
        If expression is invalid or contains disallowed operations

    Examples
    --------
    >>> evaluate_expression("len(items)", {"items": [1, 2, 3]})
    3

    >>> evaluate_expression("upper(name)", {"name": "john"})
    'JOHN'

    >>> evaluate_expression("price * quantity", {"price": 10, "quantity": 5})
    50
    """
    if not expression or not expression.strip():
        raise ExpressionError(expression, "Expression cannot be empty")

    expression = expression.strip()
    state = state or {}

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ExpressionError(expression, f"Syntax error: {e.msg}") from e

    # Validate AST is safe
    _validate_ast(tree, expression)

    return _evaluate_node(tree.body, data, state)
