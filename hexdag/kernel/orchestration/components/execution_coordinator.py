"""Execution coordinator for observer notifications and input mapping.

This module provides execution coordination functionality:

- Observer notifications during execution
- Input preparation and dependency mapping
- Input mapping transformation (including $input syntax)
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hexdag.kernel.ports.observer_manager import ObserverManager
else:
    ObserverManager = Any

from hexdag.kernel.domain.dag import NodeSpec
from hexdag.kernel.expression_parser import ALLOWED_FUNCTIONS, ExpressionError, evaluate_expression
from hexdag.kernel.logging import get_logger
from hexdag.stdlib.nodes.mapped_input import FieldExtractor

__all__ = ["ExecutionCoordinator"]

logger = get_logger(__name__)


class ExecutionCoordinator:
    """Coordinates execution context: observer notifications and input mapping.

    This component handles two responsibilities:

    1. **Observer Notifications**: Notifying observers of events during DAG execution.

    2. **Input Mapping**: Preparing input data for nodes based on their dependencies.
       Uses a smart mapping strategy:
       - No dependencies → initial input
       - Single dependency → pass through that result
       - Multiple dependencies → dict of results

    Examples
    --------
    Basic usage::

        coordinator = ExecutionCoordinator()

        # Notify observer of an event
        await coordinator.notify_observer(observer_manager, NodeStarted(...))

        # Prepare input for a node
        input_data = coordinator.prepare_node_input(
            node_spec, node_results, initial_input
        )
    """

    # ========================================================================
    # Observer Notifications (from PolicyCoordinator)
    # ========================================================================

    async def notify_observer(self, observer_manager: ObserverManager | None, event: Any) -> None:
        """Notify observer manager of an event if it exists.

        Parameters
        ----------
        observer_manager : ObserverManager | None
            Observer manager to notify (None if no observer configured)
        event : Any
            Event to send (typically NodeStarted, NodeCompleted, etc.)

        Examples
        --------
        >>> from hexdag.kernel.orchestration.events import NodeStarted
        >>> event = NodeStarted(name="my_node", wave_index=0)
        >>> await coordinator.notify_observer(observer_manager, event)  # doctest: +SKIP
        """
        if observer_manager:
            await observer_manager.notify(event)

    # ========================================================================
    # Input Mapping
    # ========================================================================

    def prepare_node_input(
        self, node_spec: NodeSpec, node_results: dict[str, Any], initial_input: Any
    ) -> Any:
        """Prepare input data for node execution with simplified data mapping.

        The mapping strategy is:
        1. **No dependencies** → initial_input (entry point)
        2. **Single dependency** → results[dependency_name] (pass-through)
        3. **Multiple dependencies** → {dep1: result1, dep2: result2, ...} (namespace)

        This approach balances simplicity (pass-through for single deps) with
        clarity (named dict for multiple deps).

        Parameters
        ----------
        node_spec : NodeSpec
            Node specification containing dependencies
        node_results : dict[str, Any]
            Results from previously executed nodes
        initial_input : Any
            Initial input data for the pipeline

        Returns
        -------
        Any
            Prepared input data for the node:
            - initial_input if no dependencies
            - dependency result if single dependency
            - dict of dependency results if multiple dependencies

        Examples
        --------
        >>> coordinator = ExecutionCoordinator()
        >>>
        >>> # No dependencies - gets initial input
        >>> # start_input = coordinator.prepare_node_input(
        >>> #     NodeSpec("start", lambda x: x.upper()),
        >>> #     node_results={},
        >>> #     initial_input="hello"
        >>> # )
        >>> # start_input == "hello"
        >>>
        >>> # Single dependency - gets that result directly
        >>> # process_input = coordinator.prepare_node_input(
        >>> #     NodeSpec("process", lambda x: x + "!", deps={"start"}),
        >>> #     node_results={"start": "HELLO"},
        >>> #     initial_input="hello"
        >>> # )
        >>> # process_input == "HELLO"

        Notes
        -----
        The multi-dependency dict preserves node names as keys, making it clear
        where each piece of data came from. This is especially useful for
        debugging and for nodes that need to treat different dependencies
        differently.

        If the node has an ``input_mapping`` in its params, the prepared input
        will be transformed according to the mapping. This supports:
        - ``$input.field`` - Reference the initial pipeline input
        - ``node_name.field`` - Reference a specific dependency's output
        """
        # Propagate skip: if ALL dependencies were skipped, return skip marker
        # so downstream nodes are auto-skipped (no need for explicit when clause)
        if node_spec.deps:
            all_skipped = all(
                isinstance(node_results.get(dep), dict)
                and node_results.get(dep, {}).get("_skipped")
                for dep in node_spec.deps
            )
            if all_skipped:
                return {"_skipped": True, "_upstream_skipped": True}

        # Prepare base input from dependencies
        if not node_spec.deps:
            base_input = initial_input
        elif len(node_spec.deps) == 1:
            dep_name = next(iter(node_spec.deps))
            base_input = node_results.get(dep_name, initial_input)
        else:
            # Multiple dependencies - preserve namespace structure
            base_input = {}
            for dep_name in node_spec.deps:
                if dep_name in node_results:
                    base_input[dep_name] = node_results[dep_name]

        # Apply input_mapping if present in node params
        input_mapping = node_spec.params.get("input_mapping") if node_spec.params else None
        if input_mapping:
            return self._apply_input_mapping(base_input, input_mapping, initial_input, node_results)

        return base_input

    def _is_expression(self, source: str) -> bool:
        """Check if a source string is an expression (contains function calls or operators).

        Parameters
        ----------
        source : str
            The source string to check

        Returns
        -------
        bool
            True if the source appears to be an expression
        """
        # Check for function call patterns (function_name followed by parenthesis)
        for func_name in ALLOWED_FUNCTIONS:
            if f"{func_name}(" in source:
                return True

        # Check for arithmetic/comparison operators (but not dots which are field paths)
        # Be careful not to match operators in simple field paths
        expression_indicators = [
            "==",
            "!=",
            "<=",
            ">=",
            " < ",
            " > ",
            " + ",
            " - ",
            " * ",
            " / ",
            " % ",
            " and ",
            " or ",
            " not ",
            " in ",
        ]
        return any(op in source for op in expression_indicators)

    def _apply_input_mapping(
        self,
        base_input: Any,
        input_mapping: dict[str, str],
        initial_input: Any,
        node_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply field mapping to transform input data.

        Supports multiple syntaxes:
        - ``$input.field`` - Extract from the initial pipeline input
        - ``node_name.field`` - Extract from a specific node's output
        - Expression syntax - Use allowed functions and operators

        Parameters
        ----------
        base_input : Any
            The prepared input from dependencies (may be single value or dict)
        input_mapping : dict[str, str]
            Mapping of {target_field: "source_path"} or {target_field: "expression"}
        initial_input : Any
            The original pipeline input (for $input references)
        node_results : dict[str, Any]
            Results from all previously executed nodes

        Returns
        -------
        dict[str, Any]
            Transformed input with mapped fields

        Examples
        --------
        >>> coordinator = ExecutionCoordinator()
        >>> mapping = {"load_id": "$input.load_id", "result": "analyzer.output"}
        >>> # This would extract load_id from initial input and result from analyzer node

        Expression examples::

            mapping = {
                "is_valid": "len(items) > 0",
                "name_upper": "upper(user.name)",
                "total": "price * quantity",
            }
        """
        result: dict[str, Any] = {}

        for target_field, source_path in input_mapping.items():
            # Guard against non-string values (e.g., from malformed YAML or !include)
            if not isinstance(source_path, str):
                logger.warning(  # type: ignore[unreachable]
                    f"input_mapping: value for '{target_field}' is "
                    f"{type(source_path).__name__}, expected str. Using value directly."
                )
                result[target_field] = source_path
                continue

            # Check if this is an expression that needs evaluation
            if self._is_expression(source_path):
                value = self._evaluate_expression(
                    source_path, base_input, initial_input, node_results
                )
            elif source_path.startswith("$input."):
                # Extract from initial pipeline input
                actual_path = source_path[7:]  # Remove "$input." prefix
                if actual_path:
                    # Has a field path like "$input.my_field"
                    if isinstance(initial_input, dict):
                        value = FieldExtractor.extract(initial_input, actual_path)
                    else:
                        # Non-dict input - wrap and extract
                        value = FieldExtractor.extract({"_root": initial_input}, "_root")
                else:
                    # Just "$input." with no field - return entire initial input
                    value = initial_input
            elif source_path == "$input":
                # Reference the entire initial input
                value = initial_input
            elif "." in source_path:
                # Check if it's a node_name.field pattern
                parts = source_path.split(".", 1)
                node_name, field_path = parts[0], parts[1]
                if node_name in node_results:
                    # Extract from specific node's result
                    value = FieldExtractor.extract(node_results[node_name], field_path)
                else:
                    # Fall back to extracting from base_input
                    value = FieldExtractor.extract(
                        base_input if isinstance(base_input, dict) else {}, source_path
                    )
            else:
                # Simple field name - extract from base_input
                value = FieldExtractor.extract(
                    base_input if isinstance(base_input, dict) else {}, source_path
                )

            if value is None:
                logger.warning(
                    f"input_mapping: '{source_path}' resolved to None for target '{target_field}'"
                )

            result[target_field] = value

        return result

    def _evaluate_expression(
        self,
        expression: str,
        base_input: Any,
        initial_input: Any,
        node_results: dict[str, Any],
    ) -> Any:
        """Evaluate an expression against available data.

        Parameters
        ----------
        expression : str
            The expression to evaluate (e.g., "len(items) > 0")
        base_input : Any
            The prepared input from dependencies
        initial_input : Any
            The original pipeline input
        node_results : dict[str, Any]
            Results from all previously executed nodes

        Returns
        -------
        Any
            The result of evaluating the expression
        """
        # Build the data context for expression evaluation
        # Merge all available data sources into a single dict
        data_context: dict[str, Any] = {}

        # Add node results
        data_context.update(node_results)

        # Add base_input (either as-is if dict, or wrapped)
        if isinstance(base_input, dict):
            data_context.update(base_input)
        elif base_input is not None:
            data_context["_input"] = base_input

        # Add initial input with $input prefix removed (accessible as 'input')
        if isinstance(initial_input, dict):
            data_context["input"] = initial_input
            # Also add initial_input fields at top level for convenience
            for key, val in initial_input.items():
                if key not in data_context:
                    data_context[key] = val
        elif initial_input is not None:
            data_context["input"] = initial_input

        try:
            # Use evaluate_expression to get the actual value, not a boolean
            return evaluate_expression(expression, data_context, {})
        except ExpressionError as e:
            logger.error(f"Expression evaluation failed for '{expression}': {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error evaluating expression '{expression}': {e}")
            return None
