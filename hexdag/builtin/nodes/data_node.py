"""Static data node for returning constant output.

.. deprecated::
    DataNode is deprecated. Use ExpressionNode directly instead.
    DataNode already delegates to ExpressionNode internally.

This module provides a DataNode factory for creating nodes that return
static data without requiring Python functions. Useful for terminal
nodes like rejection actions or static configuration.

DataNode now delegates to ExpressionNode internally, supporting both
static output and template syntax ({{variable}}).

Examples
--------
Basic usage in Python::

    from hexdag.builtin.nodes import DataNode

    node_factory = DataNode()
    node = node_factory(
        name="reject_locked",
        output={"action": "REJECTED", "reason": "Load already has winner locked"}
    )

YAML pipeline usage::

    - kind: data_node
      metadata:
        name: reject_locked
      spec:
        output:
          action: "REJECTED"
          reason: "Load already has winner locked"

With template syntax::

    - kind: data_node
      metadata:
        name: welcome_message
      spec:
        output:
          message: "Welcome {{user.name}}!"
          type: "greeting"
"""

import warnings
from typing import Any

from hexdag.core.domain.dag import NodeSpec
from hexdag.core.logging import get_logger

from .base_node_factory import BaseNodeFactory
from .expression_node import ExpressionNode, _is_template

logger = get_logger(__name__)


def _value_to_expression(value: Any) -> str:
    """Convert a value to an expression string.

    For strings without templates, wraps in quotes.
    For strings with templates, passes through.
    For other types, uses repr().

    Parameters
    ----------
    value : Any
        The value to convert

    Returns
    -------
    str
        An expression string that evaluates to the original value
    """
    if isinstance(value, str):
        if _is_template(value):
            # Template syntax - pass through for PromptTemplate rendering
            return value
        # Static string - wrap as literal expression
        return repr(value)
    if isinstance(value, bool):
        # Bool must come before int since bool is subclass of int
        return repr(value)
    if isinstance(value, (int, float)):
        return repr(value)
    if value is None:
        return "None"
    # For complex types (dict, list), use repr
    # Note: This has limitations for nested dicts with templates
    return repr(value)


class DataNode(BaseNodeFactory):
    """Static data node factory that returns constant output.

    __aliases__ declares backward-compatible names for YAML resolution.

    .. deprecated::
        DataNode is deprecated. Use ExpressionNode directly instead.
        This node already delegates to ExpressionNode internally.

    This node type eliminates the need for trivial Python functions
    that simply return static dictionaries. The output is defined
    declaratively in the YAML configuration.

    Internally delegates to ExpressionNode for unified template/expression
    handling. Supports {{variable}} template syntax for dynamic values.

    The node ignores any input data and always returns the configured
    output. Dependencies can still be specified to control execution
    order in the DAG.

    The YAML schema for this node is auto-generated from the ``__call__`` signature
    and docstrings using ``SchemaGenerator``.

    Examples
    --------
    >>> factory = DataNode()
    >>> node = factory(
    ...     name="static_response",
    ...     output={"status": "OK", "code": 200}
    ... )
    >>> node.name
    'static_response'

    With dependencies::

        >>> node = factory(
        ...     name="after_validation",
        ...     output={"result": "validated"},
        ...     deps=["validator"]
        ... )
        >>> "validator" in node.deps
        True

    With templates::

        >>> node = factory(
        ...     name="greeting",
        ...     output={"message": "Hello {{name}}!"},
        ...     deps=["user_lookup"]
        ... )
    """

    __aliases__ = ("static_node",)

    # Studio UI metadata
    _hexdag_icon = "Calculator"
    _hexdag_color = "#06b6d4"  # cyan-500 (same as ExpressionNode)

    # Schema is auto-generated from __call__ signature by SchemaGenerator

    def __call__(
        self,
        name: str,
        output: dict[str, Any],
        deps: list[str] | None = None,
        **kwargs: Any,
    ) -> NodeSpec:
        """Create a NodeSpec for a data node.

        Parameters
        ----------
        name : str
            Node name (must be unique within the pipeline)
        output : dict[str, Any]
            Output data to return. Values can be:
            - Static values (strings, numbers, bools, etc.)
            - Template strings using {{variable}} syntax
        deps : list[str] | None, optional
            List of dependency node names for execution ordering
        **kwargs : Any
            Additional parameters (when, timeout, etc.)

        Returns
        -------
        NodeSpec
            Complete node specification ready for execution

        Examples
        --------
        >>> factory = DataNode()
        >>> node = factory(
        ...     name="reject_locked",
        ...     output={"action": "REJECTED", "reason": "Load locked"}
        ... )
        >>> node.name
        'reject_locked'
        """
        warnings.warn(
            "DataNode is deprecated. Use ExpressionNode directly instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Convert output values to expressions
        expressions: dict[str, str] = {}
        for key, value in output.items():
            expressions[key] = _value_to_expression(value)

        # Delegate to ExpressionNode
        return ExpressionNode()(
            name=name,
            expressions=expressions,
            output_fields=list(output.keys()),
            deps=deps,
            **kwargs,
        )
