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


# Allowed binary operators (hoisted from inline dict)
_BINOP_MAP: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
}


# --- Individual AST node handlers ---


def _eval_constant(node: ast.AST, data: dict[str, Any], state: dict[str, Any]) -> Any:
    return node.value  # type: ignore[attr-defined]


def _eval_name(node: ast.AST, data: dict[str, Any], state: dict[str, Any]) -> Any:
    n: ast.Name = node  # type: ignore[assignment]
    if n.id == "True":
        return True
    if n.id == "False":
        return False
    if n.id == "None":
        return None
    if n.id == "state":
        return state
    return data.get(n.id)


def _eval_attribute(node: ast.AST, data: dict[str, Any], state: dict[str, Any]) -> Any:
    path = _collect_attribute_path(node)  # type: ignore[arg-type]
    return _get_value(data, state, path)


def _eval_subscript(node: ast.AST, data: dict[str, Any], state: dict[str, Any]) -> Any:
    n: ast.Subscript = node  # type: ignore[assignment]
    value = _evaluate_node(n.value, data, state)
    if isinstance(n.slice, ast.Index):  # Python 3.8 compat
        key = _evaluate_node(n.slice.value, data, state)  # type: ignore[attr-defined]
    else:
        key = _evaluate_node(n.slice, data, state)
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(key)
    if isinstance(value, (list, tuple)) and isinstance(key, int):
        return value[key] if 0 <= key < len(value) else None
    return None


def _eval_compare(node: ast.AST, data: dict[str, Any], state: dict[str, Any]) -> Any:
    n: ast.Compare = node  # type: ignore[assignment]
    left = _evaluate_node(n.left, data, state)
    result = True
    for op, comparator in zip(n.ops, n.comparators, strict=False):
        right = _evaluate_node(comparator, data, state)
        op_func = _COMPARE_OPS.get(type(op))
        if op_func is None:
            raise ExpressionError("", f"Unsupported comparison: {type(op).__name__}")
        try:
            result = result and op_func(left, right)
        except TypeError:
            return False
        left = right
    return result


def _eval_boolop(node: ast.AST, data: dict[str, Any], state: dict[str, Any]) -> Any:
    n: ast.BoolOp = node  # type: ignore[assignment]
    values = [_evaluate_node(v, data, state) for v in n.values]
    op_func = _BOOL_OPS.get(type(n.op))
    if op_func is None:
        raise ExpressionError("", f"Unsupported boolean op: {type(n.op).__name__}")
    return op_func(*values)


def _eval_unaryop(node: ast.AST, data: dict[str, Any], state: dict[str, Any]) -> Any:
    n: ast.UnaryOp = node  # type: ignore[assignment]
    operand = _evaluate_node(n.operand, data, state)
    op_func = _UNARY_OPS.get(type(n.op))
    if op_func is None:
        raise ExpressionError("", f"Unsupported unary op: {type(n.op).__name__}")
    return op_func(operand)


def _eval_ifexp(node: ast.AST, data: dict[str, Any], state: dict[str, Any]) -> Any:
    n: ast.IfExp = node  # type: ignore[assignment]
    if _evaluate_node(n.test, data, state):
        return _evaluate_node(n.body, data, state)
    return _evaluate_node(n.orelse, data, state)


def _eval_list(node: ast.AST, data: dict[str, Any], state: dict[str, Any]) -> list[Any]:
    return [_evaluate_node(elt, data, state) for elt in node.elts]  # type: ignore[attr-defined]


def _eval_tuple(node: ast.AST, data: dict[str, Any], state: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(_evaluate_node(elt, data, state) for elt in node.elts)  # type: ignore[attr-defined]


def _eval_dict(node: ast.AST, data: dict[str, Any], state: dict[str, Any]) -> dict[Any, Any]:
    n: ast.Dict = node  # type: ignore[assignment]
    keys = [_evaluate_node(k, data, state) if k else None for k in n.keys]
    values = [_evaluate_node(v, data, state) for v in n.values]
    return dict(zip(keys, values, strict=False))


def _eval_binop(node: ast.AST, data: dict[str, Any], state: dict[str, Any]) -> Any:
    n: ast.BinOp = node  # type: ignore[assignment]
    left = _evaluate_node(n.left, data, state)
    right = _evaluate_node(n.right, data, state)
    op_func = _BINOP_MAP.get(type(n.op))
    if op_func is None:
        raise ExpressionError("", f"Unsupported binary op: {type(n.op).__name__}")
    return op_func(left, right)


def _eval_call(node: ast.AST, data: dict[str, Any], state: dict[str, Any]) -> Any:
    n: ast.Call = node  # type: ignore[assignment]
    func_name = _get_function_name(n.func)
    if func_name is None or func_name not in ALLOWED_FUNCTIONS:
        raise ExpressionError("", f"Unknown or disallowed function: {func_name}")

    func = ALLOWED_FUNCTIONS[func_name]
    args = [_evaluate_node(arg, data, state) for arg in n.args]
    kwargs = {}
    for kw in n.keywords:
        if kw.arg is not None:
            kwargs[kw.arg] = _evaluate_node(kw.value, data, state)

    try:
        return func(*args, **kwargs)
    except Exception as e:
        raise ExpressionError("", f"Error calling {func_name}: {e}") from e


# Dispatch table mapping AST node types to their handlers
_NODE_HANDLERS: dict[type, Callable[[ast.AST, dict[str, Any], dict[str, Any]], Any]] = {
    ast.Constant: _eval_constant,
    ast.Name: _eval_name,
    ast.Attribute: _eval_attribute,
    ast.Subscript: _eval_subscript,
    ast.Compare: _eval_compare,
    ast.BoolOp: _eval_boolop,
    ast.UnaryOp: _eval_unaryop,
    ast.IfExp: _eval_ifexp,
    ast.List: _eval_list,
    ast.Tuple: _eval_tuple,
    ast.Dict: _eval_dict,
    ast.BinOp: _eval_binop,
    ast.Call: _eval_call,
}


def _evaluate_node(node: ast.AST, data: dict[str, Any], state: dict[str, Any]) -> Any:
    """Evaluate an AST node against data and state."""
    handler = _NODE_HANDLERS.get(type(node))
    if handler is None:
        raise ExpressionError("", f"Unsupported AST node: {type(node).__name__}")
    return handler(node, data, state)


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
