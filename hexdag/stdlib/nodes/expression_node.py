"""ExpressionNode factory for creating expression-based computation nodes.

This module provides a node type that computes values using safe AST-based
expressions, eliminating the need for boilerplate Python code when performing
calculations and transformations in YAML pipelines.

Similar to n8n's "Set Node" (Edit Fields), ExpressionNode is designed for
data transformation and computation. It also supports merge strategies for
aggregating outputs from multiple upstream dependency nodes.

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

Merge strategies for multi-dependency aggregation::

    # Collect scores from multiple nodes into a list
    - kind: expression_node
      metadata:
        name: collect_scores
      spec:
        merge_strategy: list
        extract_field: score
      dependencies: [scorer_1, scorer_2, scorer_3]

    # Calculate average score using reduce
    - kind: expression_node
      metadata:
        name: average_score
      spec:
        merge_strategy: reduce
        extract_field: score
        reducer: "statistics.mean"
      dependencies: [scorer_1, scorer_2, scorer_3]

    # Get first successful result (fallback pattern)
    - kind: expression_node
      metadata:
        name: get_result
      spec:
        merge_strategy: first
      dependencies: [primary, fallback, cache]
"""

from collections.abc import Callable
from typing import Any, Literal

from hexdag.kernel.domain.dag import NodeSpec
from hexdag.kernel.expression_parser import evaluate_expression
from hexdag.kernel.logging import get_logger
from hexdag.kernel.orchestration.prompt.template import PromptTemplate
from hexdag.kernel.resolver import resolve_function
from hexdag.stdlib.nodes.base_node_factory import BaseNodeFactory

logger = get_logger(__name__)

# Type alias for merge strategies
MergeStrategy = Literal["dict", "list", "first", "last", "reduce"]


def _is_template(expr: str) -> bool:
    """Check if expression uses template syntax (contains {{ }})."""
    return "{{" in expr and "}}" in expr


def _extract_field(value: Any, field: str | None) -> Any:
    """Extract a field from a value if specified.

    Parameters
    ----------
    value : Any
        The value to extract from
    field : str | None
        Dot-notation field path to extract (e.g., "result.score")

    Returns
    -------
    Any
        The extracted field value, or the original value if no field specified
    """
    if field is None:
        return value

    result = value
    for part in field.split("."):
        if isinstance(result, dict):
            result = result.get(part)
        elif hasattr(result, part):
            result = getattr(result, part)
        else:
            return None
    return result


def _apply_merge_strategy(
    input_data: dict[str, Any],
    strategy: MergeStrategy,
    field_path: str | None,
    reducer: Callable[[list[Any]], Any] | None,
    dep_order: list[str],
) -> Any:
    """Apply merge strategy to multi-dependency input.

    Parameters
    ----------
    input_data : dict[str, Any]
        Dict of {node_name: result} from dependencies
    strategy : MergeStrategy
        The merge strategy to apply
    field_path : str | None
        Field to extract from each result before merging (dot notation)
    reducer : Callable | None
        Reducer function for 'reduce' strategy
    dep_order : list[str]
        Ordered list of dependency names for consistent ordering

    Returns
    -------
    Any
        The merged result
    """
    # Get values in dependency order, extracting field if specified
    values = []
    for dep in dep_order:
        if dep in input_data:
            val = _extract_field(input_data[dep], field_path)
            values.append(val)

    match strategy:
        case "dict":
            # Return as-is (passthrough) or with field extraction
            if field_path:
                return {
                    dep: _extract_field(input_data[dep], field_path)
                    for dep in dep_order
                    if dep in input_data
                }
            return dict(input_data)

        case "list":
            return values

        case "first":
            for val in values:
                if val is not None:
                    return val
            return None

        case "last":
            for val in reversed(values):
                if val is not None:
                    return val
            return None

        case "reduce":
            if reducer is None:
                raise ValueError("reducer is required for 'reduce' strategy")
            # Filter out None values
            non_none_values = [v for v in values if v is not None]
            if not non_none_values:
                return None
            return reducer(non_none_values)

        case _:
            raise ValueError(f"Unknown merge strategy: {strategy}")


class ExpressionNode(BaseNodeFactory):
    """Node factory for computing values using safe AST-based expressions.

    ExpressionNode eliminates dict packing/unpacking boilerplate by:
    1. Auto-extracting input fields via input_mapping (handled by orchestrator)
    2. Evaluating chained expressions in definition order
    3. Filtering output to specified fields

    It also supports merge strategies for aggregating outputs from multiple
    upstream dependency nodes:
    - dict: Return {node_name: result, ...} (default passthrough)
    - list: Return [result1, result2, ...] in dependency order
    - first: Return first non-None result
    - last: Return last non-None result
    - reduce: Apply reducer function (e.g., statistics.mean)

    This node uses the same safe expression parser as CompositeNode, but
    returns computed values instead of routing decisions.

    See Also
    --------
    CompositeNode : For control flow (loops, conditionals)
    FunctionNode : For complex logic requiring full Python functions
    """

    # Studio UI metadata
    _hexdag_icon = "Calculator"
    _hexdag_color = "#06b6d4"  # cyan-500

    # Schema is auto-generated from __call__ signature by SchemaGenerator

    def __call__(
        self,
        name: str,
        expressions: dict[str, str] | None = None,
        input_mapping: dict[str, str] | None = None,
        output_fields: list[str] | None = None,
        deps: list[str] | None = None,
        # Merge strategy parameters
        merge_strategy: MergeStrategy | None = None,
        reducer: str | Callable[[list[Any]], Any] | None = None,
        extract_field: str | None = None,
        **kwargs: Any,
    ) -> NodeSpec:
        """Create an ExpressionNode for computing values or merging dependencies.

        Parameters
        ----------
        name : str
            Node name (unique identifier in the pipeline)
        expressions : dict[str, str] | None
            Mapping of {variable_name: expression_string}.
            Expressions are evaluated in definition order and can reference:
            - Input fields from input_mapping
            - Earlier computed variables
            - Whitelisted functions (len, max, min, Decimal, etc.)
            Optional when using merge_strategy.
        input_mapping : dict[str, str] | None
            Field extraction mapping {local_name: "source_node.field_path"}.
            Handled by the orchestrator's ExecutionCoordinator before node runs.
        output_fields : list[str] | None
            Fields to include in output dict. If None, all computed expressions
            are returned.
        deps : list[str] | None
            Dependency node names (for DAG ordering)
        merge_strategy : MergeStrategy | None
            Strategy for merging multiple dependency outputs:
            - "dict": Return {node_name: result} passthrough (default for multi-dep)
            - "list": Return [result1, result2, ...] in dependency order
            - "first": Return first non-None result
            - "last": Return last non-None result
            - "reduce": Apply reducer function to values
        reducer : str | Callable | None
            Module path (e.g., "statistics.mean") or callable for 'reduce' strategy.
            The function receives a list of values and returns a single result.
        extract_field : str | None
            Field to extract from each dependency result before merging.
            Uses dot notation (e.g., "result.score").
        **kwargs : Any
            Additional parameters passed to NodeSpec

        Returns
        -------
        NodeSpec
            Configured node specification ready for execution

        Examples
        --------
        Programmatic usage for expressions::

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

        Programmatic usage for merging::

            node = ExpressionNode()(
                name="average_score",
                merge_strategy="reduce",
                extract_field="score",
                reducer="statistics.mean",
                deps=["scorer_1", "scorer_2", "scorer_3"],
            )

        Notes
        -----
        ValueError may be raised at runtime if an expression fails to evaluate
        or references undefined variables.
        """
        # Validate: either expressions or merge_strategy must be provided
        if expressions is None and merge_strategy is None:
            raise ValueError("Either 'expressions' or 'merge_strategy' must be provided")

        # Validate: reducer is required for reduce strategy
        if merge_strategy == "reduce" and reducer is None:
            raise ValueError("'reducer' is required when merge_strategy='reduce'")

        # Store input_mapping in params for orchestrator to handle
        if input_mapping is not None:
            kwargs["input_mapping"] = input_mapping

        # Resolve reducer if it's a string module path (e.g., "statistics.mean")
        resolved_reducer: Callable[[list[Any]], Any] | None = None
        if reducer is not None:
            resolved_reducer = resolve_function(reducer) if isinstance(reducer, str) else reducer

        # Capture for closure
        _expressions = expressions or {}
        _output_fields = output_fields or list(_expressions.keys()) if _expressions else None
        _merge_strategy = merge_strategy
        _reducer = resolved_reducer
        _extract_field = extract_field
        _dep_order = list(deps or [])

        async def expression_fn(input_data: Any, **ports: Any) -> dict[str, Any] | Any:
            """Evaluate expressions and/or apply merge strategy.

            Parameters
            ----------
            input_data : Any
                Input data (typically dict after input_mapping is applied)
            **ports : Any
                Injected ports (memory, llm, etc.) - usually unused for expressions

            Returns
            -------
            dict[str, Any] | Any
                Computed values filtered to output_fields, or merged result
            """
            node_logger = logger.bind(node=name, node_type="expression_node")

            # Apply merge strategy first if specified
            if _merge_strategy is not None and isinstance(input_data, dict):
                node_logger.info(
                    "Applying merge strategy",
                    strategy=_merge_strategy,
                    extract_field=_extract_field,
                    dep_count=len(_dep_order),
                )
                merged = _apply_merge_strategy(
                    input_data=input_data,
                    strategy=_merge_strategy,
                    field_path=_extract_field,
                    reducer=_reducer,
                    dep_order=_dep_order,
                )

                # If no expressions, return merged result directly
                if not _expressions:
                    node_logger.info(
                        "Merge complete (no expressions)",
                        result_type=type(merged).__name__,
                    )
                    return {"result": merged}

                # Otherwise, make merged result available for expressions
                input_data = merged if isinstance(merged, dict) else {"merged": merged}

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
            fields_to_output = _output_fields or list(_expressions.keys())
            for field in fields_to_output:
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

        return self.create_node_with_mapping(
            name=name,
            wrapped_fn=expression_fn,
            input_schema=None,
            output_schema=None,
            deps=deps,
            **kwargs,
        )
