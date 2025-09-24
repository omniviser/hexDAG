"""Tests for LoopNode and ConditionalNode implementations (enhanced replacements)."""

import pytest

from hexai.core.bootstrap import ensure_bootstrapped
from hexai.core.domain.dag import NodeSpec
from hexai.core.registry import registry
from hexai.core.application.nodes.loop_node import (
    LoopConfig,
    ConditionalConfig,
    StopReason,
)

# Ensure registry is bootstrapped for tests
ensure_bootstrapped()


class MockEventManager:
    """Mock event manager for testing."""

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or "test-session-123"
        self.events = []

    async def emit(self, event):
        """Mock emit that just stores events."""
        self.events.append(event)


class TestLoopNode:
    """Test cases for LoopNode functionality."""

    def test_loop_node_creation(self):
        """LoopNode returns a valid NodeSpec with callable fn."""
        ensure_bootstrapped()
        loop_node = registry.get("loop_node", namespace="core")

        cfg = LoopConfig(while_condition=lambda d, s: False)
        node_spec = loop_node("test_loop_create", config=cfg)

        assert isinstance(node_spec, NodeSpec)
        assert node_spec.name == "test_loop_create"
        assert callable(node_spec.fn)

    def test_loop_node_invalid_max_iterations(self):
        """LoopNode validates max_iterations (via config and/or __call__ kwargs)."""
        ensure_bootstrapped()
        loop_node = registry.get("loop_node", namespace="core")

        with pytest.raises(ValueError, match="max_iterations must be positive"):
            LoopConfig(while_condition=lambda d, s: True, max_iterations=0)

        with pytest.raises(ValueError, match="max_iterations must be positive"):
            LoopConfig(while_condition=lambda d, s: True, max_iterations=-5)


    def test_loop_node_invalid_collect_mode(self):
        """LoopConfig validates collect_mode in __post_init__."""
        ensure_bootstrapped()

        with pytest.raises(ValueError, match="collect_mode must be one of"):
            LoopConfig(while_condition=lambda d, s: True, collect_mode="unknown")  # type: ignore[arg-type]

    def test_loop_node_reduce_requires_reducer(self):
        """Reducer must be provided when collect_mode='reduce' (validated in __post_init__)."""
        ensure_bootstrapped()

        with pytest.raises(ValueError, match="reducer is required"):
            LoopConfig(while_condition=lambda d, s: True, collect_mode="reduce", reducer=None)

    @pytest.mark.asyncio
    async def test_loop_condition_last_output(self):
        """Loop runs until condition is False; returns last output when collect_mode='last'."""
        ensure_bootstrapped()
        loop_node = registry.get("loop_node", namespace="core")

        # Condition: run while state['i'] < 3
        def while_condition(data, state):
            return state.get("i", 0) < 3

        # Body: return current 'i'
        def body_fn(data, state):
            return state.get("i", 0)

        # Update: increment i
        def on_iteration_end(state, out):
            new_state = dict(state)
            new_state["i"] = state.get("i", 0) + 1
            return new_state

        cfg = LoopConfig(
            while_condition=while_condition,
            body_fn=body_fn,
            on_iteration_end=on_iteration_end,
            init_state={"i": 0},
            collect_mode="last",
            max_iterations=10,
        )
        node_spec = loop_node("loop_last", config=cfg)

        result = await node_spec.fn({"payload": "x"})
        # Expect last output to be 2 (i: 0,1,2 -> stop when trying 3)
        assert result["result"] == 2
        assert result["metadata"]["iterations"] == 3
        assert result["metadata"]["state"]["i"] == 3

        stopped = result["metadata"]["stopped_by"]
        stopped = stopped.value if hasattr(stopped, "value") else stopped
        assert stopped == StopReason.CONDITION.value

    @pytest.mark.asyncio
    async def test_loop_collect_list_outputs(self):
        """Collects outputs per iteration when collect_mode='list'."""
        ensure_bootstrapped()
        loop_node = registry.get("loop_node", namespace="core")

        def while_condition(data, state):
            return state.get("cnt", 0) < 4

        def body_fn(data, state):
            return f"val-{state.get('cnt', 0)}"

        def on_iteration_end(state, out):
            return {"cnt": state.get("cnt", 0) + 1}

        cfg = LoopConfig(
            while_condition=while_condition,
            body_fn=body_fn,
            on_iteration_end=on_iteration_end,
            init_state={"cnt": 0},
            collect_mode="list",
            max_iterations=10,
        )
        node_spec = loop_node("loop_list", config=cfg)

        result = await node_spec.fn({"foo": "bar"})
        assert result["result"] == ["val-0", "val-1", "val-2", "val-3"]
        assert result["metadata"]["iterations"] == 4
        assert result["metadata"]["state"]["cnt"] == 4

    @pytest.mark.asyncio
    async def test_loop_reduce_outputs(self):
        """Reduces outputs with a reducer when collect_mode='reduce'."""
        ensure_bootstrapped()
        loop_node = registry.get("loop_node", namespace="core")

        def while_condition(data, state):
            return state.get("n", 0) < 5

        def body_fn(data, state):
            # produce current n
            return state.get("n", 0)

        def on_iteration_end(state, out):
            return {"n": state.get("n", 0) + 1}

        def reducer(acc, out):
            return (acc or 0) + (out or 0)

        cfg = LoopConfig(
            while_condition=while_condition,
            body_fn=body_fn,
            on_iteration_end=on_iteration_end,
            init_state={"n": 0},
            collect_mode="reduce",
            reducer=reducer,
            max_iterations=10,
        )
        node_spec = loop_node("loop_reduce", config=cfg)

        result = await node_spec.fn({"a": 1})
        # 0 + 1 + 2 + 3 + 4 = 10
        assert result["result"] == 10
        assert result["metadata"]["iterations"] == 5
        assert result["metadata"]["state"]["n"] == 5

        stopped = result["metadata"]["stopped_by"]
        stopped = stopped.value if hasattr(stopped, "value") else stopped
        assert stopped == StopReason.CONDITION.value

    @pytest.mark.asyncio
    async def test_loop_break_guard(self):
        """Breaks based on callable predicate(s)."""
        ensure_bootstrapped()
        loop_node = registry.get("loop_node", namespace="core")

        # condition always True; rely on break_if and max_iterations
        def while_condition(data, state):
            return True

        def body_fn(data, state):
            # collect iteration index
            return state.get("i", 0)

        def on_iteration_end(state, out):
            return {"i": state.get("i", 0) + 1}

        # break when i >= 3
        brk = [lambda d, s: s.get("i", 0) >= 3]

        cfg = LoopConfig(
            while_condition=while_condition,
            body_fn=body_fn,
            on_iteration_end=on_iteration_end,
            init_state={"i": 0},
            break_if=brk,
            collect_mode="list",
            max_iterations=10,
        )
        node_spec = loop_node("loop_break", config=cfg, iteration_key="iter")

        result = await node_spec.fn({"flags": {"skip": False}})
        # Iterations 0,1,2 produce outputs; on i=3 breaks (no output)
        assert result["result"] == [0, 1, 2]
        assert result["metadata"]["iterations"] == 3
        assert result["metadata"]["state"]["i"] == 3

        stopped = result["metadata"]["stopped_by"]
        stopped = stopped.value if hasattr(stopped, "value") else stopped
        assert stopped == StopReason.BREAK_GUARD.value

    @pytest.mark.asyncio
    async def test_loop_max_iterations_safety(self):
        """Stops when hitting max_iterations even if condition keeps True."""
        ensure_bootstrapped()
        loop_node = registry.get("loop_node", namespace="core")

        def while_condition(data, state):
            return True

        def body_fn(data, state):
            return "x"

        def on_iteration_end(state, out):
            c = state.get("c", 0) + 1
            return {"c": c}

        cfg = LoopConfig(
            while_condition=while_condition,
            body_fn=body_fn,
            on_iteration_end=on_iteration_end,
            init_state={"c": 0},
            collect_mode="list",
            max_iterations=2,
        )
        node_spec = loop_node("loop_safety_cap", config=cfg)

        result = await node_spec.fn({"k": "v"})
        assert result["result"] == ["x", "x"]
        assert result["metadata"]["iterations"] == 2

        stopped = result["metadata"]["stopped_by"]
        stopped = stopped.value if hasattr(stopped, "value") else stopped
        assert stopped == StopReason.LIMIT.value


class TestConditionalNode:
    """Test cases for ConditionalNode functionality (functional-only)."""

    def test_conditional_node_creation(self):
        """ConditionalNode returns a valid NodeSpec."""
        ensure_bootstrapped()
        conditional_node = registry.get("conditional_node", namespace="core")
        cfg = ConditionalConfig(branches=[{"pred": lambda d, s: True, "action": "ok"}])
        node_spec = conditional_node("test_conditional_create", config=cfg)

        assert isinstance(node_spec, NodeSpec)
        assert node_spec.name == "test_conditional_create"
        assert callable(node_spec.fn)

    @pytest.mark.asyncio
    async def test_conditional_callable_predicates(self):
        """Branches with callable predicates; picks first true."""
        ensure_bootstrapped()
        conditional_node = registry.get("conditional_node", namespace="core")

        cfg = ConditionalConfig(
            branches=[
                {"pred": lambda d, s: d.get("score", 0) >= 0.9, "action": "approve"},
                {"pred": lambda d, s: d.get("score", 0) >= 0.6, "action": "review"},
            ],
            else_action="reject",
        )
        node_spec = conditional_node("cond_callable", config=cfg)

        result = await node_spec.fn({"score": 0.75})
        assert result["result"] == "review"
        assert result["metadata"]["evaluations"] == [False, True]
        assert result["metadata"]["matched_branch"] == 1

    @pytest.mark.asyncio
    async def test_conditional_else_action(self):
        """Else branch when nothing matches."""
        ensure_bootstrapped()
        conditional_node = registry.get("conditional_node", namespace="core")

        cfg = ConditionalConfig(
            branches=[
                {"pred": lambda d, s: False, "action": "a"},
                {"pred": lambda d, s: False, "action": "b"},
            ],
            else_action="fallback",
        )
        node_spec = conditional_node("cond_else", config=cfg)

        result = await node_spec.fn({"x": {}})
        assert result["result"] == "fallback"
        assert result["metadata"]["has_else"] is True

    @pytest.mark.asyncio
    async def test_conditional_state_pass_through(self):
        """Predicates can use state passed via ports."""
        ensure_bootstrapped()
        conditional_node = registry.get("conditional_node", namespace="core")

        cfg = ConditionalConfig(
            branches=[
                {"pred": lambda d, s: s.get("flag") is True, "action": "use_state"},
                {"pred": lambda d, s: d.get("v") == 1, "action": "by_data"},
            ],
            else_action="none",
        )
        node_spec = conditional_node("cond_state", config=cfg)

        result = await node_spec.fn({"v": 0}, state={"flag": True})
        assert result["result"] == "use_state"

    def test_conditional_node_deps_mapping(self):
        """deps passed via kwargs should map to NodeSpec.deps."""
        ensure_bootstrapped()
        conditional_node = registry.get("conditional_node", namespace="core")

        cfg = ConditionalConfig(branches=[{"pred": lambda d, s: True, "action": "ok"}])
        node_spec = conditional_node(
            "cond_with_deps",
            config=cfg,
            deps=["n1", "n2"],
            custom="x",
        )

        assert isinstance(node_spec, NodeSpec)
        assert hasattr(node_spec, "deps")
        assert "n1" in node_spec.deps and "n2" in node_spec.deps
        assert node_spec.params.get("custom") == "x"
