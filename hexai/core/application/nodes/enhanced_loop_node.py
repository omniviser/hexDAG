import asyncio
import logging
from typing import Any, Callable, Iterable

from ...domain.dag import NodeSpec
from .base_node_factory import BaseNodeFactory


class EnhancedLoopNode(BaseNodeFactory):
    """
    Advanced loop control node executed entirely inside the node function.

    Features:
    - condition(data, state) -> bool controls whether an iteration should run
    - body_fn(data, state) -> Any (sync or async) executes per-iteration logic
    - on_iteration_end(state, out) -> dict updates loop state after each iteration
    - break_if / continue_if accept predicates (callable or dotted-path strings)
    - Result collection modes: 'list' | 'last' | 'reduce' (with reducer)
    - Preserves and returns loop state and metadata
    """

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
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if collect_mode not in {"list", "last", "reduce"}:
            raise ValueError("collect_mode must be one of: list | last | reduce")
        if collect_mode == "reduce" and reducer is None:
            raise ValueError("reducer is required when collect_mode='reduce'")

        break_preds = list(break_if or [])
        cont_preds = list(continue_if or [])

        async def enhanced_loop_fn(input_data: Any, **ports: Any) -> dict[str, Any]:
            log = logging.getLogger(logger_name)

            # Inline normalization (instead of a helper)
            if hasattr(input_data, "model_dump"):
                data = input_data.model_dump()
            elif isinstance(input_data, dict):
                data = dict(input_data)
            else:
                data = {"input": input_data}

            state = dict(init_state or {})

            it = 0
            outputs = [] if collect_mode == "list" else None
            last_value = None
            reduced = None

            # Local inline predicate evaluator (callable or dotted path in data)
            def eval_pred(pred: Callable[[dict, dict], bool] | str) -> bool:
                if callable(pred):
                    try:
                        return bool(pred(data, state))
                    except Exception:
                        return False
                # dotted path in data (safe, no eval)
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

            while True:
                if it >= max_iterations:
                    break

                if condition is not None:
                    try:
                        if not condition(data, state):
                            break
                    except Exception:
                        # If condition fails, stop the loop to avoid infinite cycling
                        break

                if any(eval_pred(p) for p in cont_preds):
                    it += 1
                    state[iteration_key] = it
                    continue

                if any(eval_pred(p) for p in break_preds):
                    break

                out = None
                if body_fn is not None:
                    out = body_fn(data, state)
                    if asyncio.iscoroutine(out):
                        out = await out

                if collect_mode == "list":
                    outputs.append(out)
                elif collect_mode == "last":
                    last_value = out
                else:  # reduce
                    reduced = out if reduced is None else reducer(reduced, out)

                if on_iteration_end:
                    try:
                        new_state = on_iteration_end(state, out)
                        if isinstance(new_state, dict):
                            state = new_state
                    except Exception:
                        # Ignore state update errors to keep loop robust
                        pass

                it += 1
                state[iteration_key] = it

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

        return NodeSpec(name=name, fn=enhanced_loop_fn, **kwargs)


class MultiConditionalNode(BaseNodeFactory):
    """
    Multi-branch conditional router executed inside the node function.

    - branches: list of dicts, each with:
        * pred: callable(data, state) -> bool OR dotted-path string in data (e.g. 'meta.ready')
        * action: string label for the branch, e.g. 'approve', 'review', 'reject'
    - else_action: fallback when no branch matches
    - tie_break: 'first_true' (default) — pick the first matching branch

    Output fields:
    - routing: selected action or else_action (or None if nothing matches and no else_action)
    - routing_evals: list of booleans for branch predicate evaluations
    - routing_expl: short explanation of which branch was chosen
    """

    def __call__(
        self,
        name: str,
        branches: list[dict],
        else_action: str | None = None,
        tie_break: str = "first_true",
        logger_name: str = "hexai.app.application.nodes.multi_conditional",
        **kwargs: Any,
    ) -> NodeSpec:
        if not isinstance(branches, list) or not all(isinstance(b, dict) for b in branches):
            raise ValueError("branches must be a list[dict] with keys: pred, action")
        if tie_break not in {"first_true"}:
            raise ValueError("tie_break must be 'first_true'")

        async def multi_conditional_fn(input_data: Any, **ports: Any) -> dict[str, Any]:
            log = logging.getLogger(logger_name)

            # Inline normalization (instead of a helper)
            if hasattr(input_data, "model_dump"):
                data = input_data.model_dump()
            elif isinstance(input_data, dict):
                data = dict(input_data)
            else:
                data = {"input": input_data}

            # Local inline predicate evaluator (callable or dotted path in data)
            def eval_pred(pred: Callable[[dict, dict], bool] | str) -> bool:
                if callable(pred):
                    try:
                        return bool(pred(data, {}))
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

        return NodeSpec(name=name, fn=multi_conditional_fn, **kwargs)
