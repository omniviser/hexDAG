"""LoopNode for creating loop control nodes with conditional execution.

This module provides:
- LoopNode: iterative control with a single while_condition,
state preservation, and result collection by convention.
- ConditionalNode: multi-branch router with callable predicates.

Single-style (functional) API:
- No string predicates are supported.

Conventions:
- Prefer while_condition for loop control.
- Result collection:
  - If iterating over a collection and you want all outputs → set collect_mode="list".
  - Otherwise → defaults to "last".
"""

import asyncio
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from ...domain.dag import NodeSpec
from ...registry import node
from ...registry.models import NodeSubtype
from .base_node_factory import BaseNodeFactory

logger = logging.getLogger("hexai.app.application.nodes")

CollectMode = Literal["list", "last", "reduce"]
TieBreak = Literal["first_true"]


class StopReason(str, Enum):
    """Reasons for loop termination in metadata of LoopNode"""

    CONDITION = "condition"  # while_condition returned False
    LIMIT = "limit"  # max_iterations reached
    CONDITION_ERROR = "condition_error"  # while_condition raised an exception
    BREAK_GUARD = "break_guard"  # break_if predicate triggered
    NONE = "none"  # loop did not run or ended unexpectedly


@dataclass
class LoopConfig:
    """Configuration for LoopNode (functional-only).
    - while_condition: Callable[[dict, dict], bool] — main loop predicate.
    - body_fn: Callable[[dict, dict], Any] — per-iteration body (sync or async).
    - on_iteration_end: Callable[[dict, Any], dict] — hook to update state after each iteration.
    - init_state: dict — initial mutable state dict shared across iterations.
    - collect_mode: Literal["list", "last", "reduce"] — optional override for result collection.
    - reducer: Callable[[Any, Any], Any] — required when using "reduce".
    - break_if: Iterable[Callable[[dict, dict], bool]] = If returns True after an iteration's body
    the loop termianted with StopReason.BREAK_GUARD.
    """

    # REQUIRED: no default
    while_condition: Callable[[dict, dict], bool]

    # Sensible defaults
    body_fn: Callable[[dict, dict], Any] = lambda data, state: None
    on_iteration_end: Callable[[dict, Any], dict] = lambda state, out: state
    init_state: dict = field(default_factory=dict)

    collect_mode: CollectMode = "last"
    reducer: Callable[[Any, Any], Any] | None = None

    max_iterations: int = 1
    iteration_key: str = "loop_iteration"

    break_if: Iterable[Callable[[dict, dict], bool]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.collect_mode not in ("list", "last", "reduce"):
            raise ValueError("collect_mode must be one of: list | last | reduce")
        if self.collect_mode == "reduce" and self.reducer is None:
            raise ValueError("reducer is required when collect_mode='reduce'")


@dataclass
class ConditionalConfig:
    branches: list[dict]
    else_action: str | None = None
    tie_break: TieBreak = "first_true"

    def __post_init__(self) -> None:
        if not isinstance(self.branches, list) or len(self.branches) == 0:
            raise ValueError("branches must be a non-empty list")

        if self.tie_break != "first_true":
            raise ValueError("tie_break must be 'first_true'")


def _normalize_input_data(input_data: Any) -> Any:
    """Normalize input data for loop and conditional nodes."""
    if hasattr(input_data, "model_dump"):
        data = input_data.model_dump()
    elif isinstance(input_data, dict):
        data = dict(input_data)
    else:
        data = {"input": input_data}

    return data


def _eval_break_guards(
    guards: Iterable[Callable[[dict, dict], bool]], data: dict, state: dict
) -> bool:
    """OR-semantics: return True if any guard signals to break; guard errors are ignored."""
    for idx, g in enumerate(guards):
        try:
            if g(data, state):
                return True
        except Exception as e:
            logger.debug("break_if[%d] raised; ignoring error: %s", idx, e)
    return False


def _apply_on_iteration_end(
    on_end: Callable[[dict, Any], dict] | None, state: dict, out: Any
) -> dict:
    """Run on_iteration_end; ignore errors; return possibly updated state."""
    if on_end is None:
        return state
    try:
        new_state = on_end(state, out)
        return new_state if isinstance(new_state, dict) else state
    except Exception as e:
        logger.debug("on_iteration_end failed: %s", e)
        return state


@node(name="loop_node", subtype=NodeSubtype.LOOP, namespace="core")
class LoopNode(BaseNodeFactory):
    """Advanced loop control node (functional-only).
    Key points:
    - Single controlling predicate: while_condition(data, state) -> bool (required).
    - State is preserved across iterations via the state dict (updated per iteration).
    - Result shape: {
                    "result": <list|last|reduced>,
                    "metadata": {
                        iterations, stopped_by, max_iterations, state
                        }
                    }
    - No string predicates, no eval-based resolution.
    Result collection:
    - To collect all outputs, set collect_mode="list".
    - To reduce across iterations, set collect_mode="reduce" and provide a reducer.
    - Otherwise defaults to "last".
    """

    def __init__(self) -> None:
        """Initialize LoopNode factory."""
        super().__init__()

    @staticmethod
    def _init_collector(
        mode: CollectMode, reducer: Callable[[Any, Any], Any] | None
    ) -> tuple[Any, Callable[[Any, Any], Any]]:
        """Initialize collector node."""
        if mode == "list":
            acc: Any = []

            def collect_list(a: Any, x: Any) -> Any:
                a.append(x)
                return a

            return acc, collect_list
        if mode == "last":
            acc = None

            def collect_last(_a: Any, x: Any) -> Any:
                return x

            return acc, collect_last
        if mode == "reduce":
            acc = None
            red = reducer or (lambda _a, b: b)

            def collect_reduce(a: Any, x: Any) -> Any:
                return x if a is None else red(a, x)

            return acc, collect_reduce
        raise ValueError("collect_mode must be one of: list | last | reduce")

    @staticmethod
    def _should_continue(
        while_condition: Callable[[dict, dict], bool], data: dict, state: dict
    ) -> tuple[bool, StopReason | None]:
        """Return whether condition should continue."""
        try:
            ok = bool(while_condition(data, state))
            if not ok:
                return False, StopReason.CONDITION
            return True, None
        except Exception as e:
            logger.debug("main condition raised; stop loop: %s", e)
            return False, StopReason.CONDITION_ERROR

    def __call__(
        self,
        name: str,
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
        if not isinstance(config, LoopConfig):
            raise TypeError("LoopNode requires 'config' argument of type LoopConfig")

        if not callable(config.while_condition):
            raise ValueError("LoopNode requires a callable while_condition in config")

        while_condition = config.while_condition
        body_fn = config.body_fn
        on_iteration_end = config.on_iteration_end
        init_state = config.init_state
        collect_mode = config.collect_mode
        reducer = config.reducer
        max_iterations = config.max_iterations
        iteration_key = config.iteration_key
        break_if = config.break_if

        async def loop_fn(input_data: Any, **ports: Any) -> dict[str, Any]:
            """
            Execute the enhanced loop:

            Steps per iteration:
            1) Check safety cap (max_iterations) and main condition.
            2) Run body_fn (sync/async) and collect result according to collect_mode.
            3) Update state via on_iteration_end(state, out).

            Returns a dict with:
            - Original input fields (normalized to dict).
            - "loop": metadata with iterations, final state, stop flags.
            - One of: "outputs" (list), "output" (last), or "reduced" (accumulator).
            """

            # Normalize input to dict without external helpers
            data = _normalize_input_data(input_data)

            # Local loop state
            state = dict(init_state or {})

            acc, collect_fn = self._init_collector(collect_mode, reducer)

            iteration_count = 0
            stopped_by: StopReason = StopReason.NONE

            while True:
                # Safety cap
                if iteration_count >= max_iterations:
                    stopped_by = StopReason.LIMIT
                    break

                # Main condition
                ok, reason = self._should_continue(while_condition, data, state)
                if not ok:
                    stopped_by = reason or StopReason.CONDITION
                    break

                # Body execution (supports async)
                out = body_fn(data, state)
                if asyncio.iscoroutine(out):
                    out = await out

                # Collect results
                acc = collect_fn(acc, out)

                # State update after each iteration
                state = _apply_on_iteration_end(on_iteration_end, state, out)

                # Break guard
                if _eval_break_guards(break_if, data, state):
                    stopped_by = StopReason.BREAK_GUARD
                    iteration_count += 1
                    state[iteration_key] = iteration_count
                    break

                iteration_count += 1
                state[iteration_key] = iteration_count

            # Build final result payload
            return {
                "result": acc,
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
        config: ConditionalConfig | None = None,
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
        - **kwargs: Passed through to NodeSpec (e.g., in_model, out_model, deps).
        """
        if not isinstance(config, ConditionalConfig):
            raise TypeError("ConditionalNode requires 'config' argument of type ConditionalConfig")

        branches = config.branches
        else_action = config.else_action
        tie_break = config.tie_break

        async def conditional_fn(input_data: Any, **ports: Any) -> dict[str, Any]:
            """
            Evaluate branches in order and pick the routing action.

            - Normalizes input to dict.
            - For callable predicates, passes (data, state) where state may be provided via ports.
            """

            # Normalizes input to dict
            data = _normalize_input_data(input_data)

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

            logger.info("Conditional '%s' -> routing: %s", name, chosen)
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
