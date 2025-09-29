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
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

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


@dataclass(frozen=True)
class ReduceConfig:
    reducer: Callable[[Any, Any], Any]


@dataclass
class LoopConfig:
    """Configuration for LoopNode (functional-only).
    - while_condition: Callable[[dict, dict], bool] — main loop predicate.
    - body_fn: Callable[[dict, dict], Any] — per-iteration body (sync or async).
    - on_iteration_end: Callable[[dict, Any], dict] — hook to update state after each iteration.
    - init_state: dict — initial mutable state dict shared across iterations.
    - collect_mode: Literal["list", "last", "reduce"] — optional override for result collection.
    - reduce_config: ReduceConfig — required when using "reduce".
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
    reduce_config: ReduceConfig | None = None

    max_iterations: int = 1
    iteration_key: str = "loop_iteration"

    break_if: Iterable[Callable[[dict, dict], bool]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.collect_mode not in ("list", "last", "reduce"):
            raise ValueError("collect_mode must be one of: list | last | reduce")
        if self.collect_mode == "reduce" and (
            self.reduce_config is None or self.reduce_config.reducer is None
        ):
            raise ValueError("ReduceConfig with a reducer is required when collect_mode='reduce'")


@dataclass
class ConditionalConfig:
    branches: list[dict]
    else_action: str | None = None
    tie_break: TieBreak = "first_true"

    def __post_init__(self) -> None:
        if (not self.branches) and (self.else_action is None):
            raise ValueError("branches must be a non-empty list or else_action must be provided")

        if self.tie_break != "first_true":
            raise ValueError("tie_break must be 'first_true'")


class NodeParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    in_model: Any | None = None
    out_model: Any | None = None
    deps: set[str] = Field(default_factory=set)

    @field_validator("deps", mode="before")
    @classmethod
    def _coerce_deps(cls, v: str | Collection[str] | None) -> set[str]:
        if v is None:
            return set()
        if isinstance(v, (list, tuple, set)):
            return set(v)
        if isinstance(v, str):
            return {v}
        raise ValueError("deps must be a collection of strings or a single string")


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
            logger.warning("break_if[%d] raised; ignoring error: %s", idx, e)
    return False


def _apply_on_iteration_end(
    on_end: Callable[[dict, Any], dict] | None, state: dict, out: Any
) -> dict:
    """Run on_iteration_end; ignore errors; return possibly updated state."""
    if on_end is None:
        return state
    try:
        new_state = on_end(state, out)
    except Exception as e:
        logger.warning("on_iteration_end failed: %s", e)
        return state
    if new_state is state:
        return dict(state)

    try:
        return dict(new_state)
    except Exception as e:
        logger.warning("on_iteration_end failed: %s", e)
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

    def __init__(
        self, name: str | None = None, condition: Callable[[dict, dict], bool] | None = None
    ) -> None:
        """Initialize LoopNode factory."""
        super().__init__()

        self._name: str | None = name
        self._while: Callable[[dict, dict], bool] | None = condition
        self._body: Callable[[dict, dict], Any] = lambda d, s: None
        self._on_end: Callable[[dict, Any], dict] = lambda s, _: s
        self._init_state: dict = {}
        self._collect_mode: CollectMode = "last"
        self._reduce_cfg: ReduceConfig | None = None
        self._max_iter: int = 1
        self._iter_key: str = "loop_iteration"
        self._break_if: list[Callable[[dict, dict], bool]] = []

        self._deps: set[str] = set()
        self._in_model: Any | None = None
        self._out_model: Any | None = None

    def name(self, n: str) -> "LoopNode":
        self._name = n
        return self

    def condition(self, fn: Callable[[dict, dict], bool]) -> "LoopNode":
        self._while = fn
        return self

    def do(self, fn: Callable[[dict, dict], Any]) -> "LoopNode":
        self._body = fn
        return self

    def on_iteration_end(self, fn: Callable[[dict, Any], dict]) -> "LoopNode":
        self._on_end = fn
        return self

    def init_state(self, state: dict) -> "LoopNode":
        self._init_state = dict(state or {})
        return self

    def collect_last(self) -> "LoopNode":
        self._collect_mode = "last"
        self._reduce_cfg = None
        return self

    def collect_list(self) -> "LoopNode":
        self._collect_mode = "list"
        self._reduce_cfg = None
        return self

    def collect_reduce(self, reducer: Callable[[Any, Any], Any]) -> "LoopNode":
        self._collect_mode = "reduce"
        self._reduce_cfg = ReduceConfig(reducer=reducer)
        return self

    def max_iterations(self, n: int) -> "LoopNode":
        self._max_iter = n
        return self

    def iteration_key(self, key: str) -> "LoopNode":
        self._iter_key = key
        return self

    def break_if(self, *preds: Callable[[dict, dict], bool]) -> "LoopNode":
        self._break_if.extend(preds)
        return self

    def deps(self, deps: Iterable[str]) -> "LoopNode":
        self._deps = set(deps or [])
        return self

    def in_model(self, model: Any) -> "LoopNode":
        self._in_model = model
        return self

    def out_model(self, model: Any) -> "LoopNode":
        self._out_model = model
        return self

    def build(self) -> NodeSpec:
        """Build NodeSpec from fluent builder state."""
        if not self._name:
            raise ValueError("LoopNode name is required")
        if self._while is None:
            raise ValueError("condition(...) is required")
        if self._max_iter <= 0:
            raise ValueError("max_iterations must be positive")
        if self._collect_mode == "reduce" and self._reduce_cfg is None:
            raise ValueError("ReduceConfig is required when collect_mode='reduce'")

        cfg = LoopConfig(
            while_condition=self._while,
            body_fn=self._body,
            on_iteration_end=self._on_end,
            init_state=self._init_state,
            collect_mode=self._collect_mode,
            reduce_config=self._reduce_cfg,
            max_iterations=self._max_iter,
            iteration_key=self._iter_key,
            break_if=self._break_if,
        )
        return self.__call__(
            name=self._name,
            config=cfg,
            deps=self._deps,
            in_model=self._in_model,
            out_model=self._out_model,
        )

    @staticmethod
    def _init_collector(
        mode: CollectMode, reduce_cfg: ReduceConfig | None
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
            if reduce_cfg is None:
                raise ValueError("ReduceConfig is required when collect_mode ='reduce'")
            reducer = reduce_cfg.reducer
            acc = None

            def collect_reduce(a: Any, x: Any) -> Any:
                return x if a is None else reducer(a, x)

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
            logger.warning("main condition raised; stop loop: %s", e)
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
        reducer_cfg = config.reduce_config
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

            acc, collect_fn = self._init_collector(collect_mode, reducer_cfg)

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
        try:
            params_model = NodeParams(**kwargs)
        except Exception as e:
            raise ValueError(f"Invalid node params: {e}") from e

        return NodeSpec(
            name=name,
            fn=loop_fn,
            in_model=params_model.in_model,
            out_model=params_model.out_model,
            deps=params_model.deps,
            params=params_model.model_dump(exclude_none=True),
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

    def __init__(self, name: str | None = None) -> None:
        super().__init__()
        # builder state
        self._name: str | None = name
        self._branches: list[dict[str, Any]] = []
        self._else_action: str | None = None

        # NodeSpec params
        self._deps: set[str] = set()
        self._in_model: Any | None = None
        self._out_model: Any | None = None

    def name(self, n: str) -> "ConditionalNode":
        self._name = n
        return self

    def when(
        self,
        pred: Callable[[dict, dict], bool],
        action: str,
    ) -> "ConditionalNode":
        if not callable(pred):
            raise ValueError("when(): pred must be callable")
        if not isinstance(action, str) or not action:
            raise ValueError("when(): action must be a non-empty string")
        self._branches.append({"pred": pred, "action": action})
        return self

    def otherwise(self, action: str) -> "ConditionalNode":
        if not isinstance(action, str) or not action:
            raise ValueError("otherwise(): action must be a non-empty string")
        self._else_action = action
        return self

    def deps(self, deps: Iterable[str]) -> "ConditionalNode":
        self._deps = set(deps or [])
        return self

    def in_model(self, model: Any) -> "ConditionalNode":
        self._in_model = model
        return self

    def out_model(self, model: Any) -> "ConditionalNode":
        self._out_model = model
        return self

    def build(self) -> NodeSpec:
        """Build NodeSpec from fluent builder state."""

        if not self._name:
            raise ValueError("ConditionalNode name is required")
        if not self._branches and self._else_action is None:
            raise ValueError("At least one branch (when) or otherwise(...) action is required")

        cfg = ConditionalConfig(
            branches=self._branches,
            else_action=self._else_action,
            tie_break="first_true",
        )
        return self.__call__(
            name=self._name,
            config=cfg,
            deps=self._deps,
            in_model=self._in_model,
            out_model=self._out_model,
        )

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
                    logger.warning("predicate[%d] raised; treated as False: %s", idx, e)
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

        try:
            params_model = NodeParams(**kwargs)
        except Exception as e:
            raise ValueError(f"Invalid node parameters: {e}") from e

        return NodeSpec(
            name=name,
            fn=conditional_fn,
            in_model=params_model.in_model,
            out_model=params_model.out_model,
            deps=params_model.deps,
            params=params_model.model_dump(exclude_none=True),
        )
