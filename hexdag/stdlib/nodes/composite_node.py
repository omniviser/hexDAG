"""CompositeNode - Unified control flow node for hexDAG.

This module provides a single unified node for all control flow patterns:
- while: Condition-based loop
- for-each: Collection iteration
- times: Fixed count iteration
- if-else: Single condition branch
- switch: Multi-branch conditions

All modes support two execution patterns:
- Inline body: When `body` or `body_pipeline` specified → execute within node
- Yield to downstream: When no body → yield state to downstream nodes

Examples
--------
YAML usage - while loop with inline body::

    - kind: composite_node
      metadata:
        name: retry_loop
      spec:
        mode: while
        condition: "state.attempts < 3 and not state.success"
        initial_state:
          attempts: 0
          success: false
        body: "myapp.attempt_operation"
        collect: last

YAML usage - for-each with inline nodes::

    - kind: composite_node
      metadata:
        name: process_items
      spec:
        mode: for-each
        items: "$input.items"
        concurrency: 5
        body:
          - kind: expression_node
            spec:
              expressions:
                result: "$item * 2"

YAML usage - switch for routing (no body)::

    - kind: composite_node
      metadata:
        name: router
      spec:
        mode: switch
        branches:
          - condition: "status == 'urgent'"
            action: "urgent_path"
        else_action: "default_path"
"""

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from hexdag.kernel.context import get_user_ports
from hexdag.kernel.domain.dag import NodeSpec
from hexdag.kernel.expression_parser import compile_expression, evaluate_expression
from hexdag.kernel.logging import get_logger
from hexdag.kernel.orchestration.body_executor import BodyExecutor
from hexdag.kernel.orchestration.models import NodeExecutionContext
from hexdag.kernel.utils.input_normalization import normalize_input
from hexdag.kernel.utils.node_timer import node_timer
from hexdag.stdlib.nodes.base_node_factory import BaseNodeFactory

logger = get_logger(__name__)

# Type aliases
Mode = Literal["while", "for-each", "times", "if-else", "switch"]
CollectMode = Literal["list", "last", "first", "dict", "reduce"]
ErrorHandling = Literal["fail_fast", "continue", "collect"]


class CompositeNode(BaseNodeFactory, yaml_alias="composite_node"):
    """Unified control flow node supporting while, for-each, times, if-else, switch.

    All modes support two execution patterns:
    - Inline body: When body/body_pipeline specified → execute within node
    - Yield to downstream: When no body → yield state to dependent nodes

    The YAML schema for this node is auto-generated from the ``__call__`` signature
    and docstrings using ``SchemaGenerator``.

    """

    # Studio UI metadata
    _hexdag_icon = "Layers"
    _hexdag_color = "#6366f1"  # indigo-500

    # Schema is auto-generated from __call__ signature by SchemaGenerator

    def __init__(self, base_path: Path | None = None) -> None:
        """Initialize CompositeNode factory.

        Parameters
        ----------
        base_path : Path | None
            Base directory for resolving pipeline references
        """
        super().__init__()
        self.base_path = base_path or Path.cwd()

    def __call__(
        self,
        name: str,
        mode: Mode,
        # Body specification (optional - if omitted, yields to downstream)
        body: str | list[dict[str, Any]] | Callable[..., Any] | None = None,
        body_pipeline: str | None = None,
        # Mode-specific params
        condition: str | None = None,
        items: str | None = None,
        item_var: str = "item",
        index_var: str = "index",
        count: int | None = None,
        branches: list[dict[str, Any]] | None = None,
        else_body: str | list[dict[str, Any]] | None = None,
        else_action: str | None = None,
        # Loop state management
        initial_state: dict[str, Any] | None = None,
        state_update: dict[str, str] | None = None,
        max_iterations: int = 100,
        # Concurrency (for-each, times only)
        concurrency: int = 1,
        # Result collection
        collect: CollectMode = "list",
        key_field: str | None = None,
        reducer: str | None = None,
        # Error handling
        error_handling: ErrorHandling = "fail_fast",
        # Sub-orchestrator configuration (for inline nodes and pipelines)
        max_concurrent_nodes: int = 10,
        strict_validation: bool = False,
        default_node_timeout: float | None = None,
        # Dependencies
        deps: list[str] | None = None,
        input_mapping: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> NodeSpec:
        """Create a CompositeNode NodeSpec.

        Parameters
        ----------
        name : str
            Node name (unique identifier in the pipeline)
        mode : Mode
            Control flow mode: while, for-each, times, if-else, switch
        body : str | list[dict] | Callable | None
            Body to execute. Can be:
            - Module path string (e.g., "myapp.process")
            - Callable (compiled from !py tag)
            - List of node configs (inline nodes/sub-DAG)
            - None for yield-to-downstream pattern
        body_pipeline : str | None
            Path to external pipeline YAML file
        condition : str | None
            Condition expression for while, if-else, or switch branches
        items : str | None
            Expression resolving to iterable for for-each mode
        item_var : str
            Variable name for current item (default: "item")
        index_var : str
            Variable name for current index (default: "index")
        count : int | None
            Number of iterations for times mode
        branches : list[dict] | None
            List of condition branches for switch mode
        else_body : str | list[dict] | None
            Body for else branch (if-else, switch with inline execution)
        else_action : str | None
            Action label for else branch (switch routing mode)
        initial_state : dict | None
            Initial state dict for while mode
        state_update : dict[str, str] | None
            State update expressions for while mode
        max_iterations : int
            Safety limit for while loops (default: 100)
        concurrency : int
            Max concurrent iterations for for-each/times (default: 1)
        collect : CollectMode
            Result collection mode (default: "list")
        key_field : str | None
            Field to use as key for dict collection
        reducer : str | None
            Module path to reducer function for reduce collection
        error_handling : ErrorHandling
            Error handling strategy (default: "fail_fast")
        deps : list[str] | None
            Dependency node names
        input_mapping : dict[str, str] | None
            Field extraction mapping for orchestrator
        **kwargs : Any
            Additional parameters passed to NodeSpec

        Returns
        -------
        NodeSpec
            Configured node specification ready for execution
        """
        # Store input_mapping in params for orchestrator
        if input_mapping is not None:
            kwargs["input_mapping"] = input_mapping

        # Validate mode-specific requirements
        self._validate_mode_params(
            mode=mode,
            condition=condition,
            items=items,
            count=count,
            branches=branches,
            body=body,
            body_pipeline=body_pipeline,
        )

        # Compile conditions
        compiled_condition = compile_expression(condition) if condition else None
        compiled_branches = self._compile_branches(branches) if branches else None
        compiled_state_update = (
            {k: compile_expression(v) for k, v in state_update.items()} if state_update else None
        )

        # Capture for closure
        _mode = mode
        _body = body
        _body_pipeline = body_pipeline
        _condition = compiled_condition
        _items = items
        _item_var = item_var
        _index_var = index_var
        _count = count
        _branches = compiled_branches
        _else_body = else_body
        _else_action = else_action
        _initial_state = initial_state or {}
        _state_update = compiled_state_update
        _max_iterations = max_iterations
        _concurrency = concurrency
        _collect = collect
        _key_field = key_field
        _reducer = reducer
        _error_handling = error_handling
        _base_path = self.base_path
        _max_concurrent_nodes = max_concurrent_nodes
        _strict_validation = strict_validation
        _default_node_timeout = default_node_timeout

        async def composite_fn(input_data: Any, **ports: Any) -> dict[str, Any]:
            """Execute control flow logic based on mode."""
            # Resolve ports from ContextVar if none were passed as kwargs.
            if not ports:
                ports = get_user_ports()

            node_logger = logger.bind(node=name, node_type="composite_node", mode=_mode)

            # Normalize input
            data = normalize_input(input_data)

            # Create execution context
            context = NodeExecutionContext(dag_id=name, node_id=name)

            # Create body executor with orchestrator configuration
            executor = BodyExecutor(
                base_path=_base_path,
                max_concurrent_nodes=_max_concurrent_nodes,
                strict_validation=_strict_validation,
                default_node_timeout=_default_node_timeout,
            )

            # Check if we have a body to execute (inline mode) or yield to downstream
            has_body = _body is not None or _body_pipeline is not None

            node_logger.info(
                "Starting control flow",
                has_body=has_body,
                collect=_collect,
            )

            # Dispatch to mode handler
            with node_timer() as t:
                match _mode:
                    case "while":
                        result = await _execute_while(
                            data=data,
                            condition=_condition,
                            initial_state=_initial_state,
                            state_update=_state_update,
                            max_iterations=_max_iterations,
                            body=_body,
                            body_pipeline=_body_pipeline,
                            executor=executor,
                            context=context,
                            ports=ports,
                            collect=_collect,
                            key_field=_key_field,
                            error_handling=_error_handling,
                            node_logger=node_logger,
                        )
                    case "for-each":
                        result = await _execute_foreach(
                            data=data,
                            items_expr=_items,
                            item_var=_item_var,
                            index_var=_index_var,
                            concurrency=_concurrency,
                            body=_body,
                            body_pipeline=_body_pipeline,
                            executor=executor,
                            context=context,
                            ports=ports,
                            collect=_collect,
                            key_field=_key_field,
                            error_handling=_error_handling,
                            node_logger=node_logger,
                        )
                    case "times":
                        result = await _execute_times(
                            data=data,
                            count=_count or 0,
                            index_var=_index_var,
                            concurrency=_concurrency,
                            body=_body,
                            body_pipeline=_body_pipeline,
                            executor=executor,
                            context=context,
                            ports=ports,
                            collect=_collect,
                            key_field=_key_field,
                            error_handling=_error_handling,
                            node_logger=node_logger,
                        )
                    case "if-else":
                        result = await _execute_if_else(
                            data=data,
                            condition=_condition,
                            body=_body,
                            body_pipeline=_body_pipeline,
                            else_body=_else_body,
                            executor=executor,
                            context=context,
                            ports=ports,
                            node_logger=node_logger,
                        )
                    case "switch":
                        result = await _execute_switch(
                            data=data,
                            branches=_branches,
                            else_body=_else_body,
                            else_action=_else_action,
                            executor=executor,
                            context=context,
                            ports=ports,
                            node_logger=node_logger,
                        )
                    case _:
                        raise ValueError(f"Unknown mode: {_mode}")

            result["metadata"]["duration_ms"] = t.duration_ms

            node_logger.info(
                "Control flow completed",
                duration_ms=t.duration_str,
                stopped_by=result["metadata"].get("stopped_by"),
            )

            return result

        # Preserve function metadata
        composite_fn.__name__ = f"composite_{name}"
        composite_fn.__doc__ = f"Composite node: {name} (mode={mode})"

        return self.create_node_with_mapping(
            name=name,
            wrapped_fn=composite_fn,
            input_schema=None,
            output_schema=None,
            deps=deps,
            **kwargs,
        )

    def _validate_mode_params(
        self,
        mode: Mode,
        condition: str | None,
        items: str | None,
        count: int | None,
        branches: list[dict[str, Any]] | None,
        body: Any,
        body_pipeline: str | None,
    ) -> None:
        """Validate mode-specific parameter requirements."""
        match mode:
            case "while":
                if not condition:
                    raise ValueError("while mode requires 'condition' parameter")
            case "for-each":
                if not items:
                    raise ValueError("for-each mode requires 'items' parameter")
            case "times":
                if count is None or count < 0:
                    raise ValueError("times mode requires positive 'count' parameter")
            case "if-else":
                if not condition:
                    raise ValueError("if-else mode requires 'condition' parameter")
            case "switch":
                if not branches:
                    raise ValueError("switch mode requires 'branches' parameter")
            case _:
                raise ValueError(f"Unknown mode: {mode}")

        # Validate body/body_pipeline mutual exclusivity
        if body is not None and body_pipeline is not None:
            raise ValueError("Cannot specify both 'body' and 'body_pipeline'")

    def _compile_branches(self, branches: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Compile branch conditions."""
        compiled = []
        for branch in branches:
            condition = branch.get("condition")
            if not condition:
                raise ValueError("Each branch must have a 'condition' field")

            compiled_branch = {
                "condition_fn": compile_expression(condition),
                "condition_str": condition,
                "body": branch.get("body"),
                "action": branch.get("action"),
            }
            compiled.append(compiled_branch)
        return compiled


# =============================================================================
# Mode Execution Functions
# =============================================================================


async def _execute_while(
    data: dict[str, Any],
    condition: Callable[[dict[str, Any], dict[str, Any]], bool] | None,
    initial_state: dict[str, Any],
    state_update: dict[str, Callable[..., Any]] | None,
    max_iterations: int,
    body: str | list[dict[str, Any]] | Callable[..., Any] | None,
    body_pipeline: str | None,
    executor: BodyExecutor,
    context: NodeExecutionContext,
    ports: dict[str, Any],
    collect: CollectMode,
    key_field: str | None,
    error_handling: ErrorHandling,
    node_logger: Any,
) -> dict[str, Any]:
    """Execute while loop mode."""
    state = dict(initial_state)
    results: list[Any] = []
    errors: list[dict[str, Any]] = []
    iteration = 0
    stopped_by = "exhausted"
    has_body = body is not None or body_pipeline is not None

    while iteration < max_iterations:
        # Check condition
        if condition:
            try:
                should_continue = condition(data, state)
            except Exception as e:
                node_logger.warning("Condition evaluation failed", error=str(e))
                stopped_by = "condition_error"
                break

            if not should_continue:
                stopped_by = "condition"
                break

        node_logger.debug("While iteration", iteration=iteration, state_keys=list(state.keys()))

        if has_body:
            # Execute body
            iteration_context = {
                "$index": iteration,
                "$item": None,
                "state": state,
                "iteration": iteration,
            }

            result: Any = None
            try:
                result = await executor.execute(
                    body=body,
                    body_pipeline=body_pipeline,
                    input_data=data,
                    context=context,
                    ports=ports,
                    iteration_context=iteration_context,
                )
                results.append(result)
            except Exception as e:
                if error_handling == "fail_fast":
                    raise
                errors.append({"iteration": iteration, "error": str(e)})
                if error_handling == "continue":
                    results.append(None)

            # Update state
            if state_update:
                for key, update_fn in state_update.items():
                    try:
                        # Build context for state update (includes body result)
                        update_context = {**data, "state": state, "$body": result}
                        state[key] = update_fn(update_context, state)
                    except Exception as e:
                        node_logger.warning("State update for '{}' failed: {}", key, e)
        else:
            # Yield to downstream - just track iteration
            results.append({
                "$index": iteration,
                "state": dict(state),
                "iteration": iteration,
            })

        iteration += 1
        state["iteration"] = iteration

    if iteration >= max_iterations:
        stopped_by = "limit"

    return {
        "result": _collect_results(results, collect, key_field),
        "metadata": {
            "mode": "while",
            "iterations": iteration,
            "stopped_by": stopped_by,
            "final_state": state,
            "successful": len(results) - len(errors),
            "failed": len(errors),
            "errors": errors if errors else None,
        },
    }


async def _execute_foreach(
    data: dict[str, Any],
    items_expr: str | None,
    item_var: str,
    index_var: str,
    concurrency: int,
    body: str | list[dict[str, Any]] | Callable[..., Any] | None,
    body_pipeline: str | None,
    executor: BodyExecutor,
    context: NodeExecutionContext,
    ports: dict[str, Any],
    collect: CollectMode,
    key_field: str | None,
    error_handling: ErrorHandling,
    node_logger: Any,
) -> dict[str, Any]:
    """Execute for-each loop mode."""
    # Resolve items
    items = evaluate_expression(items_expr, data, state={}) if items_expr else []

    if not hasattr(items, "__iter__"):
        raise ValueError(f"items expression must resolve to iterable, got {type(items)}")

    items_list = list(items)
    total = len(items_list)
    has_body = body is not None or body_pipeline is not None

    node_logger.info("For-each iteration", total=total, concurrency=concurrency)

    results: list[Any] = []
    errors: list[dict[str, Any]] = []

    # Use semaphore for concurrency control
    semaphore = asyncio.Semaphore(concurrency)

    async def process_item(idx: int, item: Any) -> tuple[int, Any, Exception | None]:
        async with semaphore:
            iteration_context = {
                "$item": item,
                "$index": idx,
                "$total": total,
                "$is_first": idx == 0,
                "$is_last": idx == total - 1,
                item_var: item,
                index_var: idx,
            }

            if has_body:
                try:
                    result = await executor.execute(
                        body=body,
                        body_pipeline=body_pipeline,
                        input_data=data,
                        context=context,
                        ports=ports,
                        iteration_context=iteration_context,
                    )
                    return idx, result, None
                except Exception as e:
                    return idx, None, e
            else:
                # Yield to downstream
                return idx, iteration_context, None

    # Execute all items
    tasks = [process_item(i, item) for i, item in enumerate(items_list)]
    task_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results in order
    for task_result in task_results:
        if isinstance(task_result, BaseException):
            if error_handling == "fail_fast":
                raise task_result
            errors.append({"error": str(task_result)})
            results.append(None)
        else:
            # task_result is tuple[int, Any, Exception | None]
            idx, result, error = task_result
            if error:
                if error_handling == "fail_fast":
                    raise error
                errors.append({"index": idx, "error": str(error)})
                results.append(None)
            else:
                results.append(result)

    return {
        "result": _collect_results(results, collect, key_field),
        "metadata": {
            "mode": "for-each",
            "iterations": total,
            "stopped_by": "exhausted",
            "successful": total - len(errors),
            "failed": len(errors),
            "errors": errors if errors else None,
        },
    }


async def _execute_times(
    data: dict[str, Any],
    count: int,
    index_var: str,
    concurrency: int,
    body: str | list[dict[str, Any]] | Callable[..., Any] | None,
    body_pipeline: str | None,
    executor: BodyExecutor,
    context: NodeExecutionContext,
    ports: dict[str, Any],
    collect: CollectMode,
    key_field: str | None,
    error_handling: ErrorHandling,
    node_logger: Any,
) -> dict[str, Any]:
    """Execute times loop mode."""
    has_body = body is not None or body_pipeline is not None

    node_logger.info("Times iteration", count=count, concurrency=concurrency)

    results: list[Any] = []
    errors: list[dict[str, Any]] = []

    semaphore = asyncio.Semaphore(concurrency)

    async def process_iteration(idx: int) -> tuple[int, Any, Exception | None]:
        async with semaphore:
            iteration_context = {
                "$index": idx,
                "$total": count,
                "$is_first": idx == 0,
                "$is_last": idx == count - 1,
                index_var: idx,
            }

            if has_body:
                try:
                    result = await executor.execute(
                        body=body,
                        body_pipeline=body_pipeline,
                        input_data=data,
                        context=context,
                        ports=ports,
                        iteration_context=iteration_context,
                    )
                    return idx, result, None
                except Exception as e:
                    return idx, None, e
            else:
                return idx, iteration_context, None

    tasks = [process_iteration(i) for i in range(count)]
    task_results = await asyncio.gather(*tasks, return_exceptions=True)

    for task_result in task_results:
        if isinstance(task_result, BaseException):
            if error_handling == "fail_fast":
                raise task_result
            errors.append({"error": str(task_result)})
            results.append(None)
        else:
            # task_result is tuple[int, Any, Exception | None]
            idx, result, error = task_result
            if error:
                if error_handling == "fail_fast":
                    raise error
                errors.append({"index": idx, "error": str(error)})
                results.append(None)
            else:
                results.append(result)

    return {
        "result": _collect_results(results, collect, key_field),
        "metadata": {
            "mode": "times",
            "iterations": count,
            "stopped_by": "exhausted",
            "successful": count - len(errors),
            "failed": len(errors),
            "errors": errors if errors else None,
        },
    }


async def _execute_if_else(
    data: dict[str, Any],
    condition: Callable[[dict[str, Any], dict[str, Any]], bool] | None,
    body: str | list[dict[str, Any]] | Callable[..., Any] | None,
    body_pipeline: str | None,
    else_body: str | list[dict[str, Any]] | None,
    executor: BodyExecutor,
    context: NodeExecutionContext,
    ports: dict[str, Any],
    node_logger: Any,
) -> dict[str, Any]:
    """Execute if-else mode."""
    # Evaluate condition
    condition_met = False
    if condition:
        try:
            condition_met = bool(condition(data, {}))
        except Exception as e:
            node_logger.warning("Condition evaluation failed", error=str(e))
            condition_met = False

    node_logger.debug("If-else evaluation", condition_met=condition_met)

    has_body = body is not None or body_pipeline is not None
    has_else = else_body is not None

    if condition_met and has_body:
        result = await executor.execute(
            body=body,
            body_pipeline=body_pipeline,
            input_data=data,
            context=context,
            ports=ports,
        )
        branch = "if"
    elif not condition_met and has_else:
        result = await executor.execute(
            body=else_body,
            body_pipeline=None,
            input_data=data,
            context=context,
            ports=ports,
        )
        branch = "else"
    else:
        # Yield to downstream
        result = {"condition_met": condition_met}
        branch = "if" if condition_met else "else"

    return {
        "result": result,
        "metadata": {
            "mode": "if-else",
            "condition_met": condition_met,
            "branch_taken": branch,
            "stopped_by": "condition",
        },
    }


async def _execute_switch(
    data: dict[str, Any],
    branches: list[dict[str, Any]] | None,
    else_body: str | list[dict[str, Any]] | None,
    else_action: str | None,
    executor: BodyExecutor,
    context: NodeExecutionContext,
    ports: dict[str, Any],
    node_logger: Any,
) -> dict[str, Any]:
    """Execute switch mode."""
    matched_branch: int | None = None
    matched_action: str | None = None
    evaluations: list[bool] = []

    # Evaluate branches
    if branches:
        for idx, branch in enumerate(branches):
            try:
                condition_fn = branch["condition_fn"]
                is_match = bool(condition_fn(data, {}))
            except Exception as e:
                node_logger.warning("Branch {} condition failed: {}", idx, e)
                is_match = False

            evaluations.append(is_match)

            if is_match and matched_branch is None:
                matched_branch = idx
                matched_action = branch.get("action")
                branch_body = branch.get("body")

                # Execute body if present
                if branch_body is not None:
                    result = await executor.execute(
                        body=branch_body,
                        body_pipeline=None,
                        input_data=data,
                        context=context,
                        ports=ports,
                    )
                    return {
                        "result": result,
                        "metadata": {
                            "mode": "switch",
                            "matched_branch": matched_branch,
                            "matched_action": matched_action,
                            "evaluations": evaluations,
                            "stopped_by": "branch_matched",
                        },
                    }

    # No branch matched or branch was routing-only (action)
    if matched_branch is None:
        # Use else
        if else_body is not None:
            result = await executor.execute(
                body=else_body,
                body_pipeline=None,
                input_data=data,
                context=context,
                ports=ports,
            )
            return {
                "result": result,
                "metadata": {
                    "mode": "switch",
                    "matched_branch": None,
                    "matched_action": else_action,
                    "evaluations": evaluations,
                    "stopped_by": "else",
                },
            }
        matched_action = else_action

    # Routing mode - return action label
    node_logger.info(
        "Switch routing",
        matched_branch=matched_branch,
        action=matched_action,
    )

    return {
        "result": matched_action,
        "metadata": {
            "mode": "switch",
            "matched_branch": matched_branch,
            "matched_action": matched_action,
            "evaluations": evaluations,
            "has_else": else_action is not None or else_body is not None,
            "stopped_by": "branch_matched" if matched_branch is not None else "else",
        },
    }


def _collect_results(
    results: list[Any],
    mode: CollectMode,
    key_field: str | None = None,
) -> Any:
    """Collect results according to mode."""
    if not results:
        return [] if mode == "list" else None

    match mode:
        case "list":
            return results
        case "last":
            # Find last non-None result
            for r in reversed(results):
                if r is not None:
                    return r
            return None
        case "first":
            # Find first non-None result
            for r in results:
                if r is not None:
                    return r
            return None
        case "dict":
            if not key_field:
                return dict(enumerate(results))
            result_dict = {}
            for r in results:
                if isinstance(r, dict) and key_field in r:
                    result_dict[r[key_field]] = r
            return result_dict
        case "reduce":
            # For reduce, return all results (reducer should be applied by caller)
            return results
