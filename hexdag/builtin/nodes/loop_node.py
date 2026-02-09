"""LoopNode for creating loop control nodes with conditional execution.

.. deprecated::
    LoopNode and ConditionalNode are deprecated. Use CompositeNode instead:
    - LoopNode → CompositeNode with mode='while'
    - ConditionalNode → CompositeNode with mode='switch'

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
import time
import warnings
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from hexdag.builtin.nodes.base_node_factory import BaseNodeFactory
from hexdag.core.domain.dag import NodeSpec
from hexdag.core.logging import get_logger

logger = get_logger(__name__)

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
    """Configuration for reduce-based result collection in loops."""

    reducer: Callable[[Any, Any], Any]


class NodeParams(BaseModel):
    """Parameters for node construction."""

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


class LoopNode(BaseNodeFactory):
    """Advanced loop control node (functional-only).

    .. deprecated::
        LoopNode is deprecated. Use CompositeNode with mode='while' instead.
        See hexdag.builtin.nodes.composite_node for the new unified API.

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

    # Studio UI metadata
    _hexdag_icon = "Layers"
    _hexdag_color = "#6366f1"  # indigo-500 (same as CompositeNode)

    # Explicit schema for YAML/MCP usage (builder pattern doesn't expose params well)
    _yaml_schema: dict[str, Any] = {
        "type": "object",
        "description": "Loop control node for iterative processing",
        "properties": {
            "while_condition": {
                "type": "string",
                "description": "Module path to condition function: (data, state) -> bool",
            },
            "body": {
                "type": "string",
                "description": "Module path to body function: (data, state) -> Any",
            },
            "max_iterations": {
                "type": "integer",
                "default": 100,
                "description": "Maximum number of iterations before stopping",
            },
            "collect_mode": {
                "type": "string",
                "enum": ["last", "list", "reduce"],
                "default": "last",
                "description": "How to collect results: last value, all values, or reduced",
            },
            "initial_state": {
                "type": "object",
                "default": {},
                "description": "Initial state dict passed to first iteration",
            },
            "iteration_key": {
                "type": "string",
                "default": "loop_iteration",
                "description": "Key name for current iteration number in state",
            },
        },
        "required": ["while_condition", "body"],
    }

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
        """Set the node name."""
        self._name = n
        return self

    def condition(self, fn: Callable[[dict, dict], bool] | str) -> "LoopNode":
        """Set the loop continuation condition function.

        Parameters
        ----------
        fn : Callable[[dict, dict], bool] | str
            Either a callable predicate function that takes (data, state)
            and returns bool, or a string expression like "state.iteration < 10"
            that will be compiled into a safe predicate.

        Returns
        -------
        LoopNode
            Self for method chaining.

        Examples
        --------
        Using a callable::

            node.condition(lambda d, s: s.get("iteration", 0) < 10)

        Using a string expression::

            node.condition("state.iteration < 10")
            node.condition("not done and count < max_count")
        """
        if isinstance(fn, str):
            from hexdag.core.expression_parser import compile_expression

            fn = compile_expression(fn)
        elif not callable(fn):
            raise ValueError("condition(): fn must be callable or a string expression")
        self._while = fn
        return self

    def do(self, fn: Callable[[dict, dict], Any]) -> "LoopNode":
        """Set the loop body function."""
        self._body = fn
        return self

    def on_iteration_end(self, fn: Callable[[dict, Any], dict]) -> "LoopNode":
        """Set the state update function called after each iteration."""
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
            if reduce_cfg is None or reduce_cfg.reducer is None:
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

    def build(self) -> NodeSpec:
        """Build NodeSpec with validation."""
        # Validation moved here (instead of LoopConfig)
        if not self._name:
            raise ValueError("LoopNode name is required")
        if self._while is None:
            raise ValueError("condition(...) is required")
        if self._max_iter <= 0:
            raise ValueError("max_iterations must be positive")
        if self._collect_mode not in ("list", "last", "reduce"):
            raise ValueError("collect_mode must be one of: list | last | reduce")
        if self._collect_mode == "reduce" and (
            self._reduce_cfg is None or self._reduce_cfg.reducer is None
        ):
            raise ValueError("ReduceConfig with a reducer is required when collect_mode='reduce'")

        return self(
            name=self._name,
            deps=self._deps,
            in_model=self._in_model,
            out_model=self._out_model,
        )

    def __call__(
        self,
        name: str,
        while_condition: str | Callable[[dict, dict], bool] | None = None,
        body: str | Callable[[dict, dict], Any] | None = None,
        max_iterations: int | None = None,
        collect_mode: CollectMode | None = None,
        initial_state: dict | None = None,
        iteration_key: str | None = None,
        **kwargs: Any,
    ) -> NodeSpec:
        """Builds a LoopNode NodeSpec.

        Supports two modes:
        1. Builder pattern: Use .condition(), .do(), etc. methods, then .build()
        2. YAML/direct: Pass while_condition and body as parameters

        Parameters
        ----------
        name : str
            Node name.
        while_condition : str | Callable | None
            For YAML: String expression like "state.iteration < 10".
            For builder: Set via .condition() method.
        body : str | Callable | None
            For YAML: Module path to body function (e.g., "myapp.process").
            For builder: Set via .do() method.
        max_iterations : int | None
            Safety cap to prevent infinite loops (default: 100).
        collect_mode : CollectMode | None
            How to collect results: "last", "list", or "reduce".
        initial_state : dict | None
            Initial state dict passed to first iteration.
        iteration_key : str | None
            Key name for current iteration number in state.
        **kwargs
            Passed through to NodeSpec (e.g., in_model, out_model, deps).

        Examples
        --------
        YAML usage::

            - kind: loop_node
              metadata:
                name: retry_loop
              spec:
                while_condition: "state.iteration < 3"
                body: myapp.process_item
                max_iterations: 10
                initial_state:
                  counter: 0
        """
        warnings.warn(
            "LoopNode is deprecated. Use CompositeNode with mode='while' instead. "
            "See hexdag.builtin.nodes.composite_node for the unified API.",
            DeprecationWarning,
            stacklevel=2,
        )

        from hexdag.core.expression_parser import compile_expression
        from hexdag.core.resolver import resolve

        # Determine source: YAML parameters or builder state
        if while_condition is not None:
            # YAML mode: Compile string condition
            if isinstance(while_condition, str):
                final_condition = compile_expression(while_condition)
            else:
                final_condition = while_condition
        else:
            # Builder mode: Use internal state
            if self._while is None:
                raise ValueError("while_condition is required")
            final_condition = self._while

        final_body: Callable[[dict, dict], Any]
        if body is not None:
            # YAML mode: Resolve body function from module path or use callable directly
            resolved = resolve(body) if isinstance(body, str) else body
            final_body = cast("Callable[[dict, dict], Any]", resolved)
        else:
            # Builder mode: Use internal state
            final_body = self._body

        # Use YAML params or builder state with defaults
        final_max_iter = max_iterations if max_iterations is not None else self._max_iter
        final_collect_mode = collect_mode if collect_mode is not None else self._collect_mode
        final_init_state = dict(initial_state or self._init_state or {})
        final_iter_key = iteration_key if iteration_key is not None else self._iter_key

        on_iteration_end = self._on_end
        reduce_cfg = self._reduce_cfg
        break_if = list(self._break_if or [])

        # Capture for closure
        _condition = final_condition
        _body_fn = final_body
        _max_iter = final_max_iter
        _collect_mode = final_collect_mode
        _init_state = final_init_state
        _iter_key = final_iter_key
        _on_end = on_iteration_end
        _reduce_cfg = reduce_cfg
        _break_if = break_if

        async def loop_fn(input_data: Any, **ports: Any) -> dict[str, Any]:
            """Execute the enhanced loop.

            Steps per iteration:
            1) Check safety cap (max_iterations) and main condition.
            2) Run body_fn (sync/async) and collect result according to collect_mode.
            3) Update state via on_iteration_end(state, out).

            Returns a dict with:
            - Original input fields (normalized to dict).
            - "loop": metadata with iterations, final state, stop flags.
            - One of: "outputs" (list), "output" (last), or "reduced" (accumulator).
            """
            node_logger = logger.bind(node=name, node_type="loop_node")
            start_time = time.perf_counter()

            # Log loop start
            node_logger.info(
                "Starting loop",
                max_iterations=_max_iter,
                collect_mode=_collect_mode,
            )

            # Normalize input to dict without external helpers
            data = _normalize_input_data(input_data)

            # Local loop state
            state = dict(_init_state or {})

            acc, collect_fn = self._init_collector(_collect_mode, _reduce_cfg)

            iteration_count = 0
            stopped_by: StopReason = StopReason.NONE

            while True:
                # Safety cap
                if iteration_count >= _max_iter:
                    stopped_by = StopReason.LIMIT
                    node_logger.debug(
                        "Loop reached max iterations",
                        iteration=iteration_count,
                        max_iterations=_max_iter,
                    )
                    break

                # Main condition
                ok, reason = self._should_continue(_condition, data, state)
                if not ok:
                    stopped_by = reason or StopReason.CONDITION
                    node_logger.debug(
                        "Loop condition returned False",
                        iteration=iteration_count,
                        stop_reason=stopped_by.value,
                    )
                    break

                # Log iteration start at debug level
                node_logger.debug(
                    "Loop iteration",
                    iteration=iteration_count + 1,
                    state_keys=list(state.keys()),
                )

                # Body execution (supports async)
                out = _body_fn(data, state)
                if asyncio.iscoroutine(out):
                    out = await out

                # Collect results
                acc = collect_fn(acc, out)

                # State update after each iteration
                state = _apply_on_iteration_end(_on_end, state, out)

                # Break guard
                if _eval_break_guards(_break_if, data, state):
                    stopped_by = StopReason.BREAK_GUARD
                    iteration_count += 1
                    state[_iter_key] = iteration_count
                    node_logger.debug(
                        "Break guard triggered",
                        iteration=iteration_count,
                    )
                    break

                iteration_count += 1
                state[_iter_key] = iteration_count

            # Log loop completion
            duration_ms = (time.perf_counter() - start_time) * 1000
            node_logger.info(
                "Loop completed",
                total_iterations=iteration_count,
                stopped_by=stopped_by.value,
                collect_mode=_collect_mode,
                duration_ms=f"{duration_ms:.2f}",
            )

            # Build final result payload
            return {
                "result": acc,
                "metadata": {
                    "iterations": iteration_count,
                    "stopped_by": stopped_by,
                    "max_iterations": _max_iter,
                    "state": state,
                },
            }

        # Extract framework-level parameters from kwargs
        framework = self.extract_framework_params(kwargs)

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
            deps=frozenset(params_model.deps),
            params=params_model.model_dump(exclude_none=True),
            timeout=framework["timeout"],
            max_retries=framework["max_retries"],
            when=framework["when"],
        )


class ConditionalNode(BaseNodeFactory):
    """Multi-branch conditional router (functional-only, breaking change).

    .. deprecated::
        ConditionalNode is deprecated. Use CompositeNode with mode='switch' instead.
        See hexdag.builtin.nodes.composite_node for the new unified API.

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

    # Studio UI metadata
    _hexdag_icon = "Layers"
    _hexdag_color = "#6366f1"  # indigo-500 (same as CompositeNode)

    # Explicit schema for YAML/MCP usage (builder pattern doesn't expose params well)
    _yaml_schema: dict[str, Any] = {
        "type": "object",
        "description": "Multi-branch conditional router for workflow control flow",
        "properties": {
            "branches": {
                "type": "array",
                "description": "List of condition branches evaluated in order",
                "items": {
                    "type": "object",
                    "properties": {
                        "condition": {
                            "type": "string",
                            "description": "Expression like 'node.field == value' or callable",
                        },
                        "action": {
                            "type": "string",
                            "description": "Action name to return if condition matches",
                        },
                    },
                    "required": ["condition", "action"],
                },
            },
            "else_action": {
                "type": "string",
                "description": "Default action if no branch conditions match",
            },
            "tie_break": {
                "type": "string",
                "enum": ["first_true"],
                "default": "first_true",
                "description": "Strategy for handling multiple matching branches",
            },
        },
        "required": ["branches"],
    }

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
        pred: Callable[[dict, dict], bool] | str,
        action: str,
    ) -> "ConditionalNode":
        """Add a conditional branch.

        Parameters
        ----------
        pred : Callable[[dict, dict], bool] | str
            Either a callable predicate function that takes (data, state)
            and returns bool, or a string expression like "action == 'ACCEPT'"
            that will be compiled into a safe predicate.
        action : str
            The action name to return if this branch matches.

        Returns
        -------
        ConditionalNode
            Self for method chaining.

        Examples
        --------
        Using a callable::

            node.when(lambda d, s: d.get("status") == "active", "process")

        Using a string expression::

            node.when("status == 'active'", "process")
            node.when("node.action == 'ACCEPT' and confidence > 0.8", "approve")
            node.when("state.iteration < 10", "continue")
        """
        if isinstance(pred, str):
            from hexdag.core.expression_parser import compile_expression

            pred = compile_expression(pred)
        elif not callable(pred):
            raise ValueError("when(): pred must be callable or a string expression")
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
        """Build NodeSpec with validation."""
        if not self._name:
            raise ValueError("ConditionalNode name is required")
        if not self._branches and self._else_action is None:
            raise ValueError("At least one branch (when) or otherwise(...) action is required")

        return self(
            name=self._name,
            deps=self._deps,
            in_model=self._in_model,
            out_model=self._out_model,
        )

    def __call__(
        self,
        name: str,
        branches: list[dict[str, str]] | None = None,
        else_action: str | None = None,
        tie_break: TieBreak = "first_true",
        **kwargs: Any,
    ) -> NodeSpec:
        """Builds a ConditionalNode NodeSpec.

        Supports two modes:
        1. Builder pattern: Use .when() and .otherwise() methods, then .build()
        2. YAML/direct: Pass branches and else_action as parameters

        Parameters
        ----------
        name : str
            Node name.
        branches : list[dict[str, str]] | None
            For YAML usage: List of branches with "condition" (string expression)
            and "action" fields. Example:
            [{"condition": "action == 'ACCEPT'", "action": "approve"}]
        else_action : str | None
            Optional fallback action when no branch matches.
        tie_break : TieBreak
            Branch selection strategy; only "first_true" supported.
        **kwargs
            Passed through to NodeSpec (e.g., in_model, out_model, deps).

        Examples
        --------
        YAML usage::

            - kind: conditional_node
              metadata:
                name: router
              spec:
                branches:
                  - condition: "action == 'ACCEPT'"
                    action: approve
                  - condition: "confidence < 0.5"
                    action: manual_review
                else_action: default_handler
        """
        warnings.warn(
            "ConditionalNode is deprecated. Use CompositeNode with mode='switch' instead. "
            "See hexdag.builtin.nodes.composite_node for the unified API.",
            DeprecationWarning,
            stacklevel=2,
        )

        from hexdag.core.expression_parser import compile_expression

        # Determine source of branches: builder state or YAML parameters
        if branches is not None:
            # YAML mode: Convert string conditions to compiled predicates
            compiled_branches: list[dict[str, Any]] = []
            for branch in branches:
                condition = branch.get("condition")
                action = branch.get("action")
                if not condition or not action:
                    raise ValueError(
                        f"Each branch must have 'condition' and 'action' fields. Got: {branch}"
                    )
                # Compile string expression to predicate
                pred = compile_expression(condition)
                compiled_branches.append({"pred": pred, "action": action})
            final_branches = compiled_branches
            final_else_action = else_action
        else:
            # Builder mode: Use internal state from .when() calls
            final_branches = list(self._branches or [])
            final_else_action = self._else_action

        # Capture for closure
        _branches = final_branches
        _else_action = final_else_action
        _tie_break = tie_break

        async def conditional_fn(input_data: Any, **ports: Any) -> dict[str, Any]:
            """Evaluate branches in order and pick the routing action.

            - Normalizes input to dict.
            - For callable predicates, passes (data, state) where state may be provided via ports.
            """
            node_logger = logger.bind(node=name, node_type="conditional_node")

            # Log evaluation start
            node_logger.debug(
                "Evaluating conditions",
                branch_count=len(_branches),
                has_else=_else_action is not None,
            )

            # Normalizes input to dict
            data = _normalize_input_data(input_data)

            state = ports.get("state", {}) if isinstance(ports.get("state", {}), dict) else {}

            chosen: str | None = None
            chosen_idx: int | None = None
            evaluations: list[bool] = []

            for idx, br in enumerate(_branches):
                ok = False
                try:
                    ok = bool(br["pred"](data, state))
                except Exception as e:
                    node_logger.warning(
                        "Branch predicate raised exception",
                        branch_index=idx,
                        error=str(e),
                    )
                evaluations.append(ok)
                if ok and chosen is None:
                    chosen = br["action"]
                    chosen_idx = idx
                    if _tie_break == "first_true":
                        break

            if chosen is None:
                chosen = _else_action

            result = {
                "result": chosen,
                "metadata": {
                    "matched_branch": chosen_idx,
                    "evaluations": evaluations,
                    "has_else": _else_action is not None,
                },
            }

            # Log routing decision
            node_logger.info(
                "Routing decision",
                chosen_action=chosen,
                matched_branch=chosen_idx,
                used_else=chosen_idx is None and chosen is not None,
            )
            return result

        # Extract framework-level parameters from kwargs
        framework = self.extract_framework_params(kwargs)

        try:
            params_model = NodeParams(**kwargs)
        except Exception as e:
            raise ValueError(f"Invalid node parameters: {e}") from e

        return NodeSpec(
            name=name,
            fn=conditional_fn,
            in_model=params_model.in_model,
            out_model=params_model.out_model,
            deps=frozenset(params_model.deps),
            params=params_model.model_dump(exclude_none=True),
            timeout=framework["timeout"],
            max_retries=framework["max_retries"],
            when=framework["when"],
        )
