"""Loop and Conditional nodes with advanced flow control.

This module provides factory classes that create nodes capable of:
- Loop iteration control with complex, nested conditions (EnhancedLoopNode)
- Multi-condition branching and data-driven routing (MultiConditionalNode)
- State management across iterations
- Dynamic path selection within a DAG execution
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from ...domain.dag import NodeSpec
from ...registry import node
from ...registry.models import NodeSubtype
from .base_node_factory import BaseNodeFactory

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


@node(name="enhanced_loop_node", subtype=NodeSubtype.LOOP, namespace="core")
class EnhancedLoopNode(BaseNodeFactory):
    """Advanced loop control node executed entirely inside the node function."""

    def __init__(self) -> None:
        """Initialize LoopNode factory."""
        super().__init__()

    def __call__(
        self,
        name: str,
        condition: Callable[[dict, dict], bool] | None = None,
        body_fn: Callable[[dict, dict], Any] | None = None,
        on_iteration_end: Callable[[dict, Any], dict] | None = None,
        init_state: dict | None = None,
        break_if: Iterable[Callable[[dict, dict], bool] | str] | None = None,
        continue_if: Iterable[Callable[[dict, dict], bool] | str] | None = None,
        collect_mode: str = "list",  # 'list' | 'last' | 'reduce'
        reducer: Callable[[Any, Any], Any] | None = None,
        max_iterations: int = 100,
        iteration_key: str = "loop_iteration",
        logger_name: str = "hexai.app.application.nodes.enhanced_loop",
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

        async def enhanced_loop_fn(input_data: Any, **ports: Any) -> dict[str, Any]:
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
                if condition is not None:
                    try:
                        if not condition(data, state):
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

            # Build final result payload
            result: dict[str, Any] = {
                **data,
                "loop": {
                    "iterations_completed": it,
                    "state": state,
                    "max_iterations": max_iterations,
                    "stopped_by_condition": bool(
                        condition is not None and not condition(data, state)
                    ),
                    "stopped_by_limit": it >= max_iterations,
                },
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
            fn=enhanced_loop_fn,
            in_model=in_model,
            out_model=out_model,
            deps=deps,
            params=params,
        )


@node(name="multi_conditional_node", subtype=NodeSubtype.CONDITIONAL, namespace="core")
class MultiConditionalNode(BaseNodeFactory):
    """Multi-branch conditional router executed inside the node function."""

    def __call__(self, name: str, *args: Any, **kwargs: Any) -> NodeSpec:
        branches = kwargs.pop("branches", None)
        else_action = kwargs.pop("else_action", None)
        tie_break = kwargs.pop("tie_break", "first_true")
        logger_name = kwargs.pop("logger_name", "hexai.app.application.nodes.multi_conditional")

        if not isinstance(branches, list) or not all(isinstance(b, dict) for b in (branches or [])):
            raise ValueError("branches must be a list[dict] with keys: pred, action")
        if tie_break not in {"first_true"}:
            raise ValueError("tie_break must be 'first_true'")

        async def multi_conditional_fn(input_data: Any, **ports: Any) -> dict[str, Any]:
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
                    except Exception:
                        return False
                try:
                    cur: Any = data
                    for part in str(pred).split("."):
                        if isinstance(cur, dict) and part in cur:
                            cur = cur[part]
                        else:
                            return False
                    return bool(cur)
                except Exception:
                    return False

            chosen = None
            chosen_idx = None
            evals: list[bool] = []

            for idx, br in enumerate(branches):
                pred = br.get("pred")
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
            fn=multi_conditional_fn,
            in_model=in_model,
            out_model=out_model,
            deps=deps,
            params=params,
        )
