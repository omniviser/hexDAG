"""ExpressionNode factory for creating expression-based computation nodes.

This module provides a node type that computes values using safe AST-based
expressions, eliminating the need for boilerplate Python code when performing
calculations and transformations in YAML pipelines.

Similar to n8n's "Set Node" (Edit Fields), ExpressionNode is designed for
data transformation and computation, while ConditionalNode handles routing
decisions.

Examples
--------
Basic usage in YAML::

    - kind: expression_node
      metadata:
        name: calculate_discount
      spec:
        input_mapping:
          price: "product.price"
          quantity: "order.quantity"
        expressions:
          subtotal: "price * quantity"
          discount: "0.1 if quantity > 10 else 0"
          total: "subtotal * (1 - discount)"
        output_fields: [total, discount]

Financial calculations with Decimal::

    - kind: expression_node
      metadata:
        name: calculate_counter
      spec:
        input_mapping:
          offered_rate: "extract_offer.rate"
          rate_floor: "get_context.load.rate_floor"
          counter_count: "get_context.negotiation.counter_count"
        expressions:
          discount: "Decimal('0.10') if counter_count == 0 else Decimal('0.03')"
          floor_val: "Decimal(str(rate_floor or 0))"
          counter_amount: "float(max(Decimal(str(offered_rate)) * (1 - discount), floor_val))"
        output_fields: [counter_amount]

Template syntax for string formatting::

    - kind: expression_node
      metadata:
        name: format_response
      spec:
        input_mapping:
          name: "user.name"
          quantity: "order.quantity"
        expressions:
          # Template syntax for strings (detected by {{ }})
          message: "{{name}} ordered {{quantity}} items"
          greeting: "Hello {{name}}!"
          # Expression syntax for computation
          total: "price * quantity"
        output_fields: [message, greeting, total]
"""

from typing import Any

from hexdag.builtin.nodes.base_node_factory import BaseNodeFactory
from hexdag.core.domain.dag import NodeSpec
from hexdag.core.expression_parser import evaluate_expression
from hexdag.core.logging import get_logger
from hexdag.core.orchestration.prompt.template import PromptTemplate

logger = get_logger(__name__)


def _is_template(expr: str) -> bool:
    """Check if expression uses template syntax (contains {{ }})."""
    return "{{" in expr and "}}" in expr


class ExpressionNode(BaseNodeFactory):
    """Node factory for computing values using safe AST-based expressions.

    ExpressionNode eliminates dict packing/unpacking boilerplate by:
    1. Auto-extracting input fields via input_mapping (handled by orchestrator)
    2. Evaluating chained expressions in definition order
    3. Filtering output to specified fields

    This node uses the same safe expression parser as ConditionalNode, but
    returns computed values instead of routing decisions.

    Attributes
    ----------
    _yaml_schema : dict
        JSON Schema for YAML validation and documentation

    See Also
    --------
    ConditionalNode : For routing decisions based on conditions
    FunctionNode : For complex logic requiring full Python functions
    """

    _yaml_schema: dict[str, Any] = {
        "type": "object",
        "description": "Compute values using safe AST-based expressions or template strings",
        "properties": {
            "expressions": {
                "type": "object",
                "description": "Mapping of {variable_name: expression_or_template}. "
                "Expressions are evaluated in order and can reference earlier variables. "
                "Use {{variable}} syntax for string templates, or Python expressions.",
                "additionalProperties": {"type": "string"},
            },
            "input_mapping": {
                "type": "object",
                "description": "Field extraction mapping {local_name: 'node.path'}. "
                "Handled by orchestrator before node execution.",
                "additionalProperties": {"type": "string"},
            },
            "output_fields": {
                "type": "array",
                "description": "Fields to include in output. If omitted, all computed "
                "expressions are returned.",
                "items": {"type": "string"},
            },
        },
        "required": ["expressions"],
    }

    def __call__(
        self,
        name: str,
        expressions: dict[str, str],
        input_mapping: dict[str, str] | None = None,
        output_fields: list[str] | None = None,
        deps: list[str] | None = None,
        **kwargs: Any,
    ) -> NodeSpec:
        """Create an ExpressionNode for computing values.

        Parameters
        ----------
        name : str
            Node name (unique identifier in the pipeline)
        expressions : dict[str, str]
            Mapping of {variable_name: expression_string}.
            Expressions are evaluated in definition order and can reference:
            - Input fields from input_mapping
            - Earlier computed variables
            - Whitelisted functions (len, max, min, Decimal, etc.)
        input_mapping : dict[str, str] | None
            Field extraction mapping {local_name: "source_node.field_path"}.
            Handled by the orchestrator's ExecutionCoordinator before node runs.
        output_fields : list[str] | None
            Fields to include in output dict. If None, all computed expressions
            are returned.
        deps : list[str] | None
            Dependency node names (for DAG ordering)
        **kwargs : Any
            Additional parameters passed to NodeSpec

        Returns
        -------
        NodeSpec
            Configured node specification ready for execution

        Examples
        --------
        Programmatic usage::

            node = ExpressionNode()(
                name="calculate_total",
                expressions={
                    "subtotal": "price * quantity",
                    "tax": "subtotal * 0.08",
                    "total": "subtotal + tax",
                },
                input_mapping={
                    "price": "product.price",
                    "quantity": "order.quantity",
                },
                output_fields=["total"],
                deps=["product", "order"],
            )

        Notes
        -----
        ValueError may be raised at runtime if an expression fails to evaluate
        or references undefined variables.
        """
        # Store input_mapping in params for orchestrator to handle
        if input_mapping is not None:
            kwargs["input_mapping"] = input_mapping

        # Capture for closure
        _expressions = expressions
        _output_fields = output_fields or list(expressions.keys())

        async def expression_fn(input_data: Any, **ports: Any) -> dict[str, Any]:
            """Evaluate all expressions and return computed values.

            Parameters
            ----------
            input_data : Any
                Input data (typically dict after input_mapping is applied)
            **ports : Any
                Injected ports (memory, llm, etc.) - usually unused for expressions

            Returns
            -------
            dict[str, Any]
                Computed values filtered to output_fields
            """
            node_logger = logger.bind(node=name, node_type="expression_node")
            node_logger.info(
                "Evaluating expressions",
                expression_count=len(_expressions),
                output_fields=_output_fields,
            )

            # Build context from input data
            context: dict[str, Any] = {}
            if isinstance(input_data, dict):
                context.update(input_data)
            elif input_data is not None:
                # Non-dict input - make it available as '_input'
                context["_input"] = input_data

            # Evaluate expressions in definition order (supports chaining)
            for var_name, expr in _expressions.items():
                try:
                    # Check if expression uses template syntax ({{ }})
                    if _is_template(expr):
                        # Use PromptTemplate for string rendering
                        template = PromptTemplate(expr)
                        value = template.render(**context)
                        node_logger.debug(
                            "Template rendered",
                            variable=var_name,
                            template=expr,
                            result_type=type(value).__name__,
                        )
                    else:
                        # Use expression parser for computation
                        value = evaluate_expression(expr, context, state={})
                        node_logger.debug(
                            "Expression evaluated",
                            variable=var_name,
                            expression=expr,
                            result_type=type(value).__name__,
                        )
                    context[var_name] = value
                except Exception as e:
                    node_logger.error(
                        "Expression/template evaluation failed",
                        variable=var_name,
                        expression=expr,
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    raise ValueError(
                        f"Expression '{var_name}' failed: {e}\n"
                        f"  Expression: {expr}\n"
                        f"  Available context: {list(context.keys())}"
                    ) from e

            # Filter to output_fields only
            result: dict[str, Any] = {}
            for field in _output_fields:
                if field in context:
                    result[field] = context[field]
                else:
                    node_logger.warning(
                        "Output field not found in context",
                        field=field,
                        available=list(context.keys()),
                    )

            node_logger.info(
                "Expression evaluation complete",
                output_keys=list(result.keys()),
            )
            return result

        # Preserve function metadata for debugging
        expression_fn.__name__ = f"expression_{name}"
        expression_fn.__doc__ = f"Expression node: {name}"

        # Extract framework-level parameters from kwargs
        framework = self.extract_framework_params(kwargs)

        return NodeSpec(
            name=name,
            fn=expression_fn,
            in_model=None,  # Accepts any dict input
            out_model=None,  # Returns dict output
            deps=frozenset(deps or []),
            params=kwargs,
            timeout=framework["timeout"],
            max_retries=framework["max_retries"],
            when=framework["when"],
        )
