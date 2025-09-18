"""LoopNode for creating loop control nodes with conditional execution.

This module provides the LoopNode factory that creates nodes capable of:
- Loop iteration control
- Conditional execution based on success criteria
- State management across iterations
- Dynamic routing decisions
"""

import asyncio
import logging
from collections.abc import Callable, Iterable
from typing import Any

from ...domain.dag import NodeSpec
from ...registry import node
from ...registry.models import NodeSubtype
from .base_node_factory import BaseNodeFactory


@node(name="loop_node", subtype=NodeSubtype.LOOP, namespace="core")
class LoopNode(BaseNodeFactory):
    """Advanced loop control node executed entirely inside the node function."""

    def __init__(self) -> None:
        """Initialize LoopNode factory."""
        super().__init__()

    def __call__(
        self,
        name: str,
        max_iterations: int = 3,
        iteration_key: str = "loop_iteration",
        condition: Callable[[dict, dict], bool] | None = None,
        body_fn: Callable[[dict, dict], Any] | None = None,
        on_iteration_end: Callable[[dict, Any], dict] | None = None,
        init_state: dict | None = None,
        break_if: Iterable[Callable[[dict, dict], bool] | str] | None = None,
        continue_if: Iterable[Callable[[dict, dict], bool] | str] | None = None,
        collect_mode: str = "list",  # 'list' | 'last' | 'reduce'
        reducer: Callable[[Any, Any], Any] | None = None,
        success_condition: Callable[[dict], bool] | None = None,
        logger_name: str = "hexai.app.application.nodes.loop_node",
        **kwargs: Any,
    ) -> NodeSpec:
        # Validate configuration
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if collect_mode not in {"list", "last", "reduce"}:
            raise ValueError("collect_mode must be one of: list | last | reduce")
        if collect_mode == "reduce" and reducer is None:
            raise ValueError("reducer is required when collect_mode='reduce'")

        break_preds = list(break_if or [])
        cont_preds = list(continue_if or [])

        def legacy_condition(data: dict, state: dict) -> bool:
            if success_condition is None:
                return True
            try:
                return not bool(success_condition(data))
            except Exception:  # noqa: BLE001
                return True

        main_condition = condition if condition is not None else legacy_condition

        async def loop_fn(input_data: Any, **ports: Any) -> dict[str, Any]:
            """
            Execute the enhanced loop:

            Steps per iteration:
            1) Check safety cap (max_iterations) and main condition.
            2) Evaluate continue/break guards.
            3) Run body_fn (sync/async) and collect result according to collect_mode.
            4) Update state via on_iteration_end(state, out).

            Returns a dict with:
            - Original input fields (normalized to dict).
            - "loop": metadata with iterations, final state, stop flags.
            - One of: "outputs" (list), "output" (last), or "reduced" (accumulator).
            """
            log = logging.getLogger(logger_name)

            # Normalize input to dict without external helpers
            if hasattr(input_data, "model_dump"):
                data = input_data.model_dump()
            elif isinstance(input_data, dict):
                data = dict(input_data)
            else:
                data = {"input": input_data}

            # Local loop state
            state = dict(init_state or {})

            def eval_pred(pred: Callable[[dict, dict], bool] | str) -> bool:
                if callable(pred):
                    try:
                        return bool(pred(data, state))
                    except Exception:  # noqa: BLE001
                        return False
                try:
                    cur: Any = data
                    for part in str(pred).split("."):
                        if isinstance(cur, dict) and part in cur:
                            cur = cur[part]
                        else:
                            return False
                    return bool(cur)
                except Exception:  # noqa: BLE001
                    return False

            mode = collect_mode

            def _noop_reducer(_: Any, b: Any) -> Any:
                return b

            outputs: list[Any] | None = None
            last_value: Any | None = None
            reduced: Any | None = None
            _reducer: Callable[[Any, Any], Any] = _noop_reducer  # always callable

            if mode == "list":
                outputs = []
            elif mode == "last":
                last_value = None
            elif mode == "reduce":
                if reducer is None:
                    raise ValueError("reducer is required when collect_mode='reduce'")
                _reducer = reducer
                reduced = None
            else:
                raise ValueError("collect_mode must be one of: list | last | reduce")
            it = 0

            while True:
                # Safety cap
                if it >= max_iterations:
                    break

                # Main condition
                if main_condition is not None:
                    try:
                        if not main_condition(data, state):
                            break
                    except Exception:
                        # Defensive: stop if condition raises
                        break

                # Continue guards (skip current iteration)
                if any(eval_pred(p) for p in cont_preds):
                    it += 1
                    state[iteration_key] = it
                    continue

                # Break guards
                if any(eval_pred(p) for p in break_preds):
                    break

                # Body execution (supports async)
                out = None
                if body_fn is not None:
                    out = body_fn(data, state)
                    if asyncio.iscoroutine(out):
                        out = await out

                # Collect results
                if mode == "list":
                    if outputs is None:
                        raise RuntimeError("internal error: outputs not initialized")
                    outputs.append(out)
                elif mode == "last":
                    last_value = out
                elif mode == "reduce":
                    reduced = out if reduced is None else _reducer(reduced, out)

                # State update after each iteration
                if on_iteration_end:
                    try:
                        new_state = on_iteration_end(state, out)
                        if isinstance(new_state, dict):
                            state = new_state
                    except Exception as e:  # nosec B110
                        log.debug("on_iteration_end failed: %s", e)

                it += 1
                state[iteration_key] = it

            legacy_success = False
            if success_condition is not None:
                try:
                    legacy_success = bool(success_condition(data))
                except Exception:  # noqa: BLE001
                    legacy_success = False
            legacy_should_continue = it < max_iterations and not legacy_success

            # Build final result payload
            result: dict[str, Any] = {
                **data,
                "loop": {
                    "iterations_completed": it,
                    "state": state,
                    "max_iterations": max_iterations,
                    "stopped_by_limit": it >= max_iterations,
                },
                "iterations_completed": it,
                "success_criteria_met": legacy_success,
                "success": legacy_success,
                "should_continue": legacy_should_continue,
            }
            if collect_mode == "list":
                result["outputs"] = outputs
            elif collect_mode == "last":
                result["output"] = last_value
            else:
                result["reduced"] = reduced

            log.info("EnhancedLoop '%s' finished: iterations=%d, mode=%s", name, it, collect_mode)
            return result

        # Map DirectedGraph-related arguments to NodeSpec fields
        deps = set(kwargs.pop("deps", []) or [])
        in_model = kwargs.pop("in_model", None)
        out_model = kwargs.pop("out_model", None)
        params = kwargs  # any remaining kwargs go into NodeSpec.params

        return NodeSpec(
            name=name,
            fn=loop_fn,
            in_model=in_model,
            out_model=out_model,
            deps=deps,
            params=params,
        )


@node(name="conditional_node", subtype=NodeSubtype.CONDITIONAL, namespace="core")
class ConditionalNode(BaseNodeFactory):
    """Multi-branch conditional router executed inside the node function."""

    def __call__(
        self,
        name: str,
        condition_key: str | None = None,
        true_action: str = "continue",
        false_action: str = "proceed",
        # Enhanced API:
        branches: list[dict] | None = None,
        else_action: str | None = None,
        tie_break: str = "first_true",
        logger_name: str = "hexai.app.application.nodes.conditional_node",
        **kwargs: Any,
    ) -> NodeSpec:
        if branches is None and condition_key is not None:
            branches = [{"pred": condition_key, "action": true_action}]
            else_action = false_action

        if not isinstance(branches, list) or not all(isinstance(b, dict) for b in (branches or [])):
            raise ValueError("branches must be a list[dict] with keys: pred, action")
        if tie_break not in {"first_true"}:
            raise ValueError("tie_break must be 'first_true'")

        async def conditional_fn(input_data: Any, **ports: Any) -> dict[str, Any]:
            """
            Evaluate branches in order and pick the routing action.

            - Normalizes input to dict.
            - For callable predicates, passes (data, state) where state may be provided via ports.
            - For string predicates, resolves dotted paths against data safely (no eval).
            """
            log = logging.getLogger(logger_name)

            # Normalize input to dict
            if hasattr(input_data, "model_dump"):
                data = input_data.model_dump()
            elif isinstance(input_data, dict):
                data = dict(input_data)
            else:
                data = {"input": input_data}

            # Optional state for predicates (if upstream provided it as a port)
            state = ports.get("state", {}) if isinstance(ports.get("state", {}), dict) else {}

            # Local predicate evaluator (callable or dotted path)
            def eval_pred(pred: Callable[[dict, dict], bool] | str) -> bool:
                if callable(pred):
                    try:
                        return bool(pred(data, state))
                    except Exception:  # noqa: BLE001
                        return False
                try:
                    cur: Any = data
                    for part in str(pred).split("."):
                        if isinstance(cur, dict) and part in cur:
                            cur = cur[part]
                        else:
                            return False
                    return bool(cur)
                except Exception:  # noqa: BLE001
                    return False

            chosen = None
            chosen_idx = None
            evals: list[bool] = []

            for idx, br in enumerate(branches):
                pred = br.get("pred")
                if pred is None:
                    evals.append(False)
                    continue
                ok = eval_pred(pred)
                evals.append(ok)
                if ok and chosen is None:
                    chosen = br.get("action")
                    chosen_idx = idx
                    if tie_break == "first_true":
                        break

            if chosen is None:
                chosen = else_action

            result = {
                **data,
                "action": chosen,
                "routing": chosen,
                "routing_evals": evals,
                "routing_expl": (
                    f"branch[{chosen_idx}] selected"
                    if chosen is not None and chosen_idx is not None
                    else (
                        "else selected"
                        if chosen is not None
                        else "no branch matched and no else_action"
                    )
                ),
            }

            log.info("MultiConditional '%s' -> routing: %s", name, chosen)
            return result

        # Map DirectedGraph-related arguments to NodeSpec fields
        deps = set(kwargs.pop("deps", []) or [])
        in_model = kwargs.pop("in_model", None)
        out_model = kwargs.pop("out_model", None)
        params = kwargs  # any remaining kwargs go into NodeSpec.params

        return NodeSpec(
            name=name,
            fn=conditional_fn,
            in_model=in_model,
            out_model=out_model,
            deps=deps,
            params=params,
        )
