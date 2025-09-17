"""Tests for EnhancedLoopNode and MultiConditionalNode implementations."""

import pytest

from hexai.core.bootstrap import ensure_bootstrapped
from hexai.core.domain.dag import NodeSpec
from hexai.core.registry import registry

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


class TestEnhancedLoopNode:
    """Test cases for EnhancedLoopNode functionality."""

    def test_enhanced_loop_node_creation(self):
        """EnhancedLoopNode returns a valid NodeSpec with callable fn."""
        ensure_bootstrapped()
        enhanced_loop = registry.get("enhanced_loop_node", namespace="core")

        node_spec = enhanced_loop(
            "test_enh_loop_create",
            max_iterations=3,
        )

        assert isinstance(node_spec, NodeSpec)
        assert node_spec.name == "test_enh_loop_create"
        assert callable(node_spec.fn)

    def test_enhanced_loop_node_invalid_max_iterations(self):
        """EnhancedLoopNode validates max_iterations."""
        ensure_bootstrapped()
        enhanced_loop = registry.get("enhanced_loop_node", namespace="core")

        with pytest.raises(ValueError, match="max_iterations must be positive"):
            enhanced_loop("bad_loop_max0", max_iterations=0)

        with pytest.raises(ValueError, match="max_iterations must be positive"):
            enhanced_loop("bad_loop_maxneg", max_iterations=-5)

    def test_enhanced_loop_node_invalid_collect_mode(self):
        """EnhancedLoopNode validates collect_mode."""
        ensure_bootstrapped()
        enhanced_loop = registry.get("enhanced_loop_node", namespace="core")

        with pytest.raises(ValueError, match="collect_mode must be one of"):
            enhanced_loop("bad_collect_mode", collect_mode="unknown")

    def test_enhanced_loop_node_reduce_requires_reducer(self):
        """Reducer must be provided when collect_mode='reduce'."""
        ensure_bootstrapped()
        enhanced_loop = registry.get("enhanced_loop_node", namespace="core")

        with pytest.raises(ValueError, match="reducer is required"):
            enhanced_loop("reduce_without_reducer", collect_mode="reduce")

    @pytest.mark.asyncio
    async def test_enhanced_loop_simple_condition_and_last_output(self):
        """Loop runs until condition is False; returns last output when collect_mode='last'."""
        ensure_bootstrapped()
        enhanced_loop = registry.get("enhanced_loop_node", namespace="core")

        # Condition: run while state['i'] < 3
        def condition(data, state):
            return state.get("i", 0) < 3

        # Body: return current 'i'
        def body_fn(data, state):
            return state.get("i", 0)

        # Update: increment i
        def on_iteration_end(state, out):
            new_state = dict(state)
            new_state["i"] = state.get("i", 0) + 1
            return new_state

        node_spec = enhanced_loop(
            "loop_last",
            condition=condition,
            body_fn=body_fn,
            on_iteration_end=on_iteration_end,
            init_state={"i": 0},
            collect_mode="last",
            max_iterations=10,
        )

        result = await node_spec.fn({"payload": "x"})
        # Expect last output to be 2 (i: 0,1,2 -> stop when trying 3)
        assert result["output"] == 2
        assert result["loop"]["iterations_completed"] == 3
        assert result["loop"]["state"]["i"] == 3
        assert result["loop"]["stopped_by_condition"] is True
        assert result["loop"]["stopped_by_limit"] is False

    @pytest.mark.asyncio
    async def test_enhanced_loop_collect_list_outputs(self):
        """Collects outputs per iteration when collect_mode='list'."""
        ensure_bootstrapped()
        enhanced_loop = registry.get("enhanced_loop_node", namespace="core")

        def condition(data, state):
            return state.get("cnt", 0) < 4

        def body_fn(data, state):
            return f"val-{state.get('cnt', 0)}"

        def on_iteration_end(state, out):
            return {"cnt": state.get("cnt", 0) + 1}

        node_spec = enhanced_loop(
            "loop_list",
            condition=condition,
            body_fn=body_fn,
            on_iteration_end=on_iteration_end,
            init_state={"cnt": 0},
            collect_mode="list",
            max_iterations=10,
        )

        result = await node_spec.fn({"foo": "bar"})
        assert result["outputs"] == ["val-0", "val-1", "val-2", "val-3"]
        assert result["loop"]["iterations_completed"] == 4
        assert result["loop"]["state"]["cnt"] == 4

    @pytest.mark.asyncio
    async def test_enhanced_loop_reduce_outputs(self):
        """Reduces outputs with a reducer when collect_mode='reduce'."""
        ensure_bootstrapped()
        enhanced_loop = registry.get("enhanced_loop_node", namespace="core")

        def condition(data, state):
            return state.get("n", 0) < 5

        def body_fn(data, state):
            # produce current n
            return state.get("n", 0)

        def on_iteration_end(state, out):
            return {"n": state.get("n", 0) + 1}

        def reducer(acc, out):
            return (acc or 0) + (out or 0)

        node_spec = enhanced_loop(
            "loop_reduce",
            condition=condition,
            body_fn=body_fn,
            on_iteration_end=on_iteration_end,
            init_state={"n": 0},
            collect_mode="reduce",
            reducer=reducer,
            max_iterations=10,
        )

        result = await node_spec.fn({"a": 1})
        # 0 + 1 + 2 + 3 + 4 = 10
        assert result["reduced"] == 10
        assert result["loop"]["iterations_completed"] == 5
        assert result["loop"]["state"]["n"] == 5

    @pytest.mark.asyncio
    async def test_enhanced_loop_break_and_continue_guards(self):
        """Breaks and continues based on predicates (dotted path and callable)."""
        ensure_bootstrapped()
        enhanced_loop = registry.get("enhanced_loop_node", namespace="core")

        # condition always True; rely on break/continue and max_iterations
        def condition(data, state):
            return True

        def body_fn(data, state):
            # collect iteration index
            return state.get("i", 0)

        def on_iteration_end(state, out):
            return {"i": state.get("i", 0) + 1}

        # continue when "flags.skip" is True (dotted path)
        continue_if = ["flags.skip"]
        # break when callable returns True for i >= 3
        break_if = [lambda d, s: s.get("i", 0) >= 3]

        node_spec = enhanced_loop(
            "loop_break_continue",
            condition=condition,
            body_fn=body_fn,
            on_iteration_end=on_iteration_end,
            init_state={"i": 0},
            continue_if=continue_if,
            break_if=break_if,
            collect_mode="list",
            iteration_key="iter",
            max_iterations=10,
        )

        result = await node_spec.fn({"flags": {"skip": False}})
        # Iterations 0,1,2 produce outputs; on i=3 breaks (no output)
        assert result["outputs"] == [0, 1, 2]
        assert result["loop"]["iterations_completed"] == 3
        assert result["loop"]["state"]["i"] == 3
        assert result["loop"]["stopped_by_limit"] is False

    @pytest.mark.asyncio
    async def test_enhanced_loop_max_iterations_safety(self):
        """Stops when hitting max_iterations even if condition keeps True."""
        ensure_bootstrapped()
        enhanced_loop = registry.get("enhanced_loop_node", namespace="core")

        def condition(data, state):
            return True

        def body_fn(data, state):
            return "x"

        def on_iteration_end(state, out):
            c = state.get("c", 0) + 1
            return {"c": c}

        node_spec = enhanced_loop(
            "loop_safety_cap",
            condition=condition,
            body_fn=body_fn,
            on_iteration_end=on_iteration_end,
            init_state={"c": 0},
            collect_mode="list",
            max_iterations=2,
        )

        result = await node_spec.fn({"k": "v"})
        assert result["outputs"] == ["x", "x"]
        assert result["loop"]["stopped_by_limit"] is True
        assert result["loop"]["iterations_completed"] == 2

    def test_enhanced_loop_node_deps_mapping(self):
        """deps passed via kwargs should map to NodeSpec.deps (not params)."""
        ensure_bootstrapped()
        enhanced_loop = registry.get("enhanced_loop_node", namespace="core")
        node_spec = enhanced_loop(
            "loop_with_deps",
            max_iterations=1,
            deps=["a", "b"],
            some_param="x",
        )
        assert isinstance(node_spec, NodeSpec)
        assert "some_param" in node_spec.params
        # Depending on NodeSpec implementation, deps may be a set
        assert hasattr(node_spec, "deps")
        assert "a" in node_spec.deps and "b" in node_spec.deps


class TestMultiConditionalNode:
    """Test cases for MultiConditionalNode functionality."""

    def test_multi_conditional_node_creation(self):
        """MultiConditionalNode returns a valid NodeSpec."""
        ensure_bootstrapped()
        mc = registry.get("multi_conditional_node", namespace="core")

        node_spec = mc(
            "mc_create",
            branches=[
                {"pred": lambda d, s: True, "action": "ok"},
            ],
            else_action="fallback",
        )

        assert isinstance(node_spec, NodeSpec)
        assert node_spec.name == "mc_create"
        assert callable(node_spec.fn)

    def test_multi_conditional_node_invalid_branches(self):
        """Validates branches argument shape."""
        ensure_bootstrapped()
        mc = registry.get("multi_conditional_node", namespace="core")

        with pytest.raises(ValueError, match="branches must be a list\\[dict\\]"):
            mc("mc_invalid", branches="not-a-list")  # type: ignore[arg-type]

    def test_multi_conditional_node_invalid_tie_break(self):
        """Validates tie_break value."""
        ensure_bootstrapped()
        mc = registry.get("multi_conditional_node", namespace="core")

        with pytest.raises(ValueError, match="tie_break must be 'first_true'"):
            mc("mc_tie", branches=[], tie_break="random")

    @pytest.mark.asyncio
    async def test_multi_conditional_callable_predicates(self):
        """Selects the first branch whose callable predicate is True."""
        ensure_bootstrapped()
        mc = registry.get("multi_conditional_node", namespace="core")

        node_spec = mc(
            "mc_callable",
            branches=[
                {"pred": lambda d, s: d.get("score", 0) >= 0.9, "action": "approve"},
                {"pred": lambda d, s: d.get("score", 0) >= 0.6, "action": "review"},
            ],
            else_action="reject",
        )

        result = await node_spec.fn({"score": 0.75})
        assert result["routing"] == "review"
        assert result["routing_evals"] == [False, True]
        assert "branch[" in result["routing_expl"]

    @pytest.mark.asyncio
    async def test_multi_conditional_dotted_path_predicates(self):
        """Evaluates string predicates as dotted paths against data."""
        ensure_bootstrapped()
        mc = registry.get("multi_conditional_node", namespace="core")

        node_spec = mc(
            "mc_dotted",
            branches=[
                {"pred": "meta.ready", "action": "go"},
                {"pred": "flags.approve", "action": "approve"},
            ],
            else_action="hold",
        )

        result = await node_spec.fn({"meta": {"ready": True}, "flags": {"approve": False}})
        assert result["routing"] == "go"
        assert result["routing_evals"][0] is True

    @pytest.mark.asyncio
    async def test_multi_conditional_else_action(self):
        """Falls back to else_action when no branch matches."""
        ensure_bootstrapped()
        mc = registry.get("multi_conditional_node", namespace="core")

        node_spec = mc(
            "mc_else",
            branches=[
                {"pred": lambda d, s: False, "action": "a"},
                {"pred": "x.y", "action": "b"},
            ],
            else_action="fallback",
        )

        result = await node_spec.fn({"x": {}})
        assert result["routing"] == "fallback"
        assert "else selected" in result["routing_expl"]

    @pytest.mark.asyncio
    async def test_multi_conditional_state_is_passed_to_predicates(self):
        """Callable predicates receive (data, state) and can use state."""
        ensure_bootstrapped()
        mc = registry.get("multi_conditional_node", namespace="core")

        node_spec = mc(
            "mc_state",
            branches=[
                {"pred": lambda d, s: s.get("flag") is True, "action": "use_state"},
                {"pred": lambda d, s: d.get("v") == 1, "action": "by_data"},
            ],
            else_action="none",
        )

        result = await node_spec.fn({"v": 0}, state={"flag": True})
        assert result["routing"] == "use_state"

    def test_multi_conditional_node_deps_mapping(self):
        """deps passed via kwargs should map to NodeSpec.deps."""
        ensure_bootstrapped()
        mc = registry.get("multi_conditional_node", namespace="core")

        node_spec = mc(
            "mc_with_deps",
            branches=[{"pred": lambda d, s: True, "action": "ok"}],
            deps=["n1", "n2"],
            custom="x",
        )

        assert isinstance(node_spec, NodeSpec)
        assert hasattr(node_spec, "deps")
        assert "n1" in node_spec.deps and "n2" in node_spec.deps
        assert node_spec.params.get("custom") == "x"
