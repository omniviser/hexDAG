"""LoopNode for creating loop control nodes with conditional execution.

This module provides:
- LoopNode: iterative control with a single while_condition, functional guards,
state preservation, and result collection by convention.
- ConditionalNode: multi-branch router with callable predicates.

Single-style (functional) API:
- All conditions and guards are callables.
- No string predicates are supported.

Conventions:
- Prefer while_condition for loop control.
- Result collection:
  - If reducer provided and collect_mode not set → defaults to "reduce".
  - If iterating over a collection and you want all outputs → set collect_mode="list".
  - Otherwise → defaults to "last".
"""

import asyncio
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

from ...domain.dag import NodeSpec
from ...registry import node
from ...registry.models import NodeSubtype
from .base_node_factory import BaseNodeFactory

CollectMode = Literal["list", "last", "reduce"]


@dataclass
class LoopConfig:
    """Configuration for LoopNode (functional-only).
    - while_condition: Callable[[dict, dict], bool] — main loop predicate.
    - body_fn: Callable[[dict, dict], Any] — per-iteration body (sync or async).
    - on_iteration_end: Callable[[dict, Any], dict] — hook to update state after each iteration.
    - init_state: dict — initial mutable state dict shared across iterations.
    - break_if / continue_if: Iterable[Callable[[dict, dict], bool]] — guard predicates.
    - collect_mode: Literal["list", "last", "reduce"] — optional override for result collection.
    - reducer: Callable[[Any, Any], Any] — required when using "reduce".
    - logger_name: str — logger to use for debug/info.
    """

    while_condition: Callable[[dict, dict], bool] | None = None
    body_fn: Callable[[dict, dict], Any] | None = None
    on_iteration_end: Callable[[dict, Any], dict] | None = None
    init_state: dict | None = None

    # Guards (functional only)
    break_if: Iterable[Callable[[dict, dict], bool]] | None = None
    continue_if: Iterable[Callable[[dict, dict], bool]] | None = None

    collect_mode: CollectMode | None = None
    reducer: Callable[[Any, Any], Any] | None = None
    logger_name: str = "hexai.app.application.nodes.loop_node"


@node(name="loop_node", subtype=NodeSubtype.LOOP, namespace="core")
class LoopNode(BaseNodeFactory):
    """Advanced loop control node (functional-only).
    Key points:
    - Single controlling predicate: while_condition(data, state) -> bool (required).
    - Guards: break_if / continue_if as callables returning bool.
    - State is preserved across iterations via the state dict (updated per iteration).
    - Result shape: {
                    "result": <list|last|reduced>,
                    "metadata": {
                        iterations, stopped_by, max_iterations, state
                        }
                    }
    - No string predicates, no eval-based resolution.
    Result collection:
    - If collect_mode is None:
      - If reducer is provided -> "reduce"
      - Otherwise -> "last"
    - To collect all iteration outputs, set collect_mode="list".
    """

    def __init__(self) -> None:
        """Initialize LoopNode factory."""
        super().__init__()

    def __call__(
        self,
        name: str,
        max_iterations: int = 3,
        iteration_key: str = "loop_iteration",
        config: LoopConfig | None = None,
        **kwargs: Any,
    ) -> NodeSpec:
        """Builds a LoopNode NodeSpec.
        Parameters:
            - name: Node name.
            - max_iterations: Safety cap to prevent infinite loops (must be > 0).
            - iteration_key: Key under which the current iteration is stored in state.
            - config: LoopConfig with while_condition and optional behaviors.
            - **kwargs: Passed through to NodeSpec (e.g., in_model, out_model, deps).
        """
        if config is None or not callable(config.while_condition):
            raise ValueError("LoopNode requires a callable while_condition in config")

        while_condition = config.while_condition
        body_fn = config.body_fn
        on_iteration_end = config.on_iteration_end
        init_state = config.init_state
        break_if = config.break_if
        continue_if = config.continue_if
        collect_mode = config.collect_mode
        reducer = config.reducer
        logger_name = config.logger_name

        logger = logging.getLogger(logger_name)

        if collect_mode is None:
            collect_mode = "reduce" if reducer is not None else "last"

        # Validate configuration
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if collect_mode not in {"list", "last", "reduce"}:
            raise ValueError("collect_mode must be one of: list | last | reduce")
        if collect_mode == "reduce" and reducer is None:
            raise ValueError("reducer is required when collect_mode='reduce'")

        break_preds = [p for p in (break_if or ()) if p is not None]
        continue_predicates = [p for p in (continue_if or ()) if p is not None]

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

            # Normalize input to dict without external helpers
            if hasattr(input_data, "model_dump"):
                data = input_data.model_dump()
            elif isinstance(input_data, dict):
                data = dict(input_data)
            else:
                data = {"input": input_data}

            # Local loop state
            state = dict(init_state or {})
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
                _reducer = reducer or _noop_reducer
                reduced = None
            else:
                raise ValueError("collect_mode must be one of: list | last | reduce")

            iteration_count = 0
            stopped_by: str | None = None

            while True:
                # Safety cap
                if iteration_count >= max_iterations:
                    stopped_by = "limit"
                    break

                # Main condition
                while_cond: Callable[[dict, dict], bool] = while_condition
                try:
                    if not while_cond(data, state):
                        stopped_by = "condition"
                        break
                except Exception as e:
                    # Defensive: stop if condition raises
                    logger.debug("main condition raised; stopping loop: %s", e)
                    stopped_by = "condition_error"
                    break

                # Continue guards (skip current iteration)
                try:
                    if any(p(data, state) for p in continue_predicates):
                        iteration_count += 1
                        state[iteration_key] = iteration_count
                        continue
                except Exception as e:
                    logger.debug("continue guards evaluation failed: %s", e)

                # Break guards
                try:
                    if any(p(data, state) for p in break_preds):
                        stopped_by = "break_guard"
                        break
                except Exception as e:
                    logger.debug("break guards evaluation failed: %s", e)

                # Body execution (supports async)
                out = None
                if body_fn is not None:
                    fn: Callable[[dict, dict], Any] = body_fn
                    try:
                        out = fn(data, state)
                        if asyncio.iscoroutine(out):
                            out = await out
                    except Exception as e:
                        logger.debug("body_fn raised; output set to None %s", e)
                        out = None
                else:
                    logger.debug("body_fn is not callable")

                # Collect results
                if mode == "list":
                    outputs.append(out)  # type: ignore[union-attr]
                elif mode == "last":
                    last_value = out
                elif mode == "reduce":
                    reduced = out if reduced is None else _reducer(reduced, out)

                # State update after each iteration
                if on_iteration_end is not None:
                    try:
                        it_end: Callable[[dict, Any], dict] = on_iteration_end
                        new_state = it_end(state, out)
                        if isinstance(new_state, dict):
                            state = new_state
                    except Exception as e:  # nosec B110
                        logger.debug("on_iteration_end failed: %s", e)
                else:
                    logger.debug(
                        "on_iteration_end is not callable; on_iteration_end returned: %s",
                        on_iteration_end,
                    )

                iteration_count += 1
                state[iteration_key] = iteration_count

            if mode == "list":
                payload = outputs
            elif mode == "last":
                payload = last_value
            else:
                payload = reduced

            # Build final result payload
            return {
                "result": payload,
                "metadata": {
                    "iterations": iteration_count,
                    "stopped_by": stopped_by,
                    "max_iterations": max_iterations,
                    "state": state,
                },
            }

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
    """Multi-branch conditional router (functional-only, breaking change).
    API:
    - branches: list of {"pred": Callable[[dict, dict], bool], "action": str}
    - else_action: str | None — fallback action if no branch matches.
    - tie_break: currently only "first_true" is supported.
    Return:
        {
            "result": <action | None>,
            "metadata": {
            "matched_branch": <int | None>,
            "evaluations": <list[bool]>,
            "has_else":
            }
        }

    Notes:
    - Functional-only predicates (no strings, no eval).
    - Input is normalized to dict internally; original input is not echoed back.
    """

    def __call__(
        self,
        name: str,
        branches: list[dict] | None = None,
        else_action: str | None = None,
        tie_break: str = "first_true",
        logger_name: str = "hexai.app.application.nodes.conditional_node",
        **kwargs: Any,
    ) -> NodeSpec:
        """Builds a ConditionalNode NodeSpec (functional-only).

        Parameters:

        - name: Node name.
        - branches: Non-empty list of branches; each branch requires:
                - "pred": Callable[[dict, dict], bool]
                - "action": non-empty str
        - else_action: Optional fallback action when no branch matches.
        - tie_break: Branch selection strategy; only "first_true" supported.
        - logger_name: Logger to use.
        - **kwargs: Passed through to NodeSpec (e.g., in_model, out_model, deps).
        """
        if not branches or not isinstance(branches, list):
            raise ValueError("branches must be a non-empty list of dicts")

        if not all(
            isinstance(b, dict)
            and callable(b.get("pred"))
            and isinstance(b.get("action"), str)
            and b.get("action")
            for b in branches
        ):
            raise ValueError(
                "branch must be {'pred': Callable[[dict, dict], bool], 'action': non-empty str}"
            )
        if tie_break != "first_true":
            raise ValueError("tie_break must be 'first_true'")

        logger = logging.getLogger(logger_name)

        async def conditional_fn(input_data: Any, **ports: Any) -> dict[str, Any]:
            """
            Evaluate branches in order and pick the routing action.

            - Normalizes input to dict.
            - For callable predicates, passes (data, state) where state may be provided via ports.
            """

            # Normalizes input to dict
            if hasattr(input_data, "model_dump"):
                data = input_data.model_dump()
            elif isinstance(input_data, dict):
                data = dict(input_data)
            else:
                data = {"input": input_data}

            state = ports.get("state", {}) if isinstance(ports.get("state", {}), dict) else {}

            chosen: str | None = None
            chosen_idx: int | None = None
            evaluations: list[bool] = []

            for idx, br in enumerate(branches):
                ok = False
                try:
                    ok = bool(br["pred"](data, state))
                except Exception as e:
                    logger.debug("predicate[%d] raised; treated as False: %s", idx, e)
                evaluations.append(ok)
                if ok and chosen is None:
                    chosen = br["action"]
                    chosen_idx = idx
                    if tie_break == "first_true":
                        break

            if chosen is None:
                chosen = else_action

            result = {
                "result": chosen,
                "metadata": {
                    "matched_branch": chosen_idx,
                    "evaluations": evaluations,
                    "has_else": else_action is not None,
                },
            }

            logger.info("MultiConditional '%s' -> routing: %s", name, chosen)
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
