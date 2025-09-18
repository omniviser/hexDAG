"""Tests for LoopNode and ConditionalNode implementations (enhanced replacements)."""

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


class TestLoopNode:
    """Test cases for LoopNode functionality (enhanced)."""

    def test_loop_node_creation(self):
        """LoopNode returns a valid NodeSpec with callable fn."""
        ensure_bootstrapped()
        loop_node = registry.get("loop_node", namespace="core")

        node_spec = loop_node("test_loop_create", max_iterations=3)

        assert isinstance(node_spec, NodeSpec)
        assert node_spec.name == "test_loop_create"
        assert callable(node_spec.fn)

    def test_loop_node_invalid_max_iterations(self):
        """LoopNode validates max_iterations."""
        ensure_bootstrapped()
        loop_node = registry.get("loop_node", namespace="core")

        with pytest.raises(ValueError, match="max_iterations must be positive"):
            loop_node("bad_loop_max0", max_iterations=0)

        with pytest.raises(ValueError, match="max_iterations must be positive"):
            loop_node("bad_loop_maxneg", max_iterations=-5)

    def test_loop_node_invalid_collect_mode(self):
        """LoopNode validates collect_mode."""
        ensure_bootstrapped()
        loop_node = registry.get("loop_node", namespace="core")

        with pytest.raises(ValueError, match="collect_mode must be one of"):
            loop_node("bad_collect_mode", collect_mode="unknown")

    def test_loop_node_reduce_requires_reducer(self):
        """Reducer must be provided when collect_mode='reduce'."""
        ensure_bootstrapped()
        loop_node = registry.get("loop_node", namespace="core")

        with pytest.raises(ValueError, match="reducer is required"):
            loop_node("reduce_without_reducer", collect_mode="reduce")

    @pytest.mark.asyncio
    async def test_loop_condition_last_output(self):
        """Loop runs until condition is False; returns last output when collect_mode='last'."""
        ensure_bootstrapped()
        loop_node = registry.get("loop_node", namespace="core")

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

        node_spec = loop_node(
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
        # In now-legacy mapping we don't expose stopped_by_condition; only stopped_by_limit
        assert result["loop"]["stopped_by_limit"] is False

    @pytest.mark.asyncio
    async def test_loop_collect_list_outputs(self):
        """Collects outputs per iteration when collect_mode='list'."""
        ensure_bootstrapped()
        loop_node = registry.get("loop_node", namespace="core")

        def condition(data, state):
            return state.get("cnt", 0) < 4

        def body_fn(data, state):
            return f"val-{state.get('cnt', 0)}"

        def on_iteration_end(state, out):
            return {"cnt": state.get("cnt", 0) + 1}

        node_spec = loop_node(
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
        assert result["foo"] == "bar"  # original data preserved

    @pytest.mark.asyncio
    async def test_loop_reduce_outputs(self):
        """Reduces outputs with a reducer when collect_mode='reduce'."""
        ensure_bootstrapped()
        loop_node = registry.get("loop_node", namespace="core")

        def condition(data, state):
            return state.get("n", 0) < 5

        def body_fn(data, state):
            # produce current n
            return state.get("n", 0)

        def on_iteration_end(state, out):
            return {"n": state.get("n", 0) + 1}

        def reducer(acc, out):
            return (acc or 0) + (out or 0)

        node_spec = loop_node(
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
    async def test_loop_break_and_continue_guards(self):
        """Breaks and continues based on predicates (dotted path and callable)."""
        ensure_bootstrapped()
        loop_node = registry.get("loop_node", namespace="core")

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

        node_spec = loop_node(
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
    async def test_loop_max_iterations_safety(self):
        """Stops when hitting max_iterations even if condition keeps True."""
        ensure_bootstrapped()
        loop_node = registry.get("loop_node", namespace="core")

        def condition(data, state):
            return True

        def body_fn(data, state):
            return "x"

        def on_iteration_end(state, out):
            c = state.get("c", 0) + 1
            return {"c": c}

        node_spec = loop_node(
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

    @pytest.mark.asyncio
    async def test_loop_legacy_success_and_should_continue(self):
        """Validates legacy fields success and should_continue when using success_condition."""
        ensure_bootstrapped()
        loop_node = registry.get("loop_node", namespace="core")

        def success_condition(data):
            return data.get("score", 0) > 0.8

        node_spec = loop_node("loop_legacy", max_iterations=5, success_condition=success_condition)

        # success => True; loop should end quickly (no guarantee of iterations without body)
        result = await node_spec.fn({"score": 0.9})
        assert result["success"] is True
        assert result["should_continue"] is False

        result = await node_spec.fn({"score": 0.1})
        assert result["success"] is False
        assert isinstance(result["should_continue"], bool)


class TestConditionalNode:
    """Test cases for ConditionalNode functionality (enhanced)."""

    def test_conditional_node_creation(self):
        """ConditionalNode returns a valid NodeSpec."""
        ensure_bootstrapped()
        conditional_node = registry.get("conditional_node", namespace="core")
        node_spec = conditional_node(
            "test_conditional_create",
            condition_key="should_continue",
            true_action="retry",
            false_action="proceed",
        )

        assert isinstance(node_spec, NodeSpec)
        assert node_spec.name == "test_conditional_create"
        assert callable(node_spec.fn)

    @pytest.mark.asyncio
    async def test_conditional_true_branch_via_legacy_sugar(self):
        """Legacy sugar: condition_key + true_action/false_action."""
        ensure_bootstrapped()
        conditional_node = registry.get("conditional_node", namespace="core")
        node_spec = conditional_node(
            "cond_true",
            condition_key="should_continue",
            true_action="retry",
            false_action="proceed",
        )

        result = await node_spec.fn({"should_continue": True, "confidence": "low", "extra_data": "test"})
        # Structure and values
        assert result["action"] == "retry"
        assert result["routing"] == "retry"
        assert "routing_evals" in result
        assert "routing_expl" in result
        # Original data preserved
        assert result["confidence"] == "low"
        assert result["extra_data"] == "test"

    @pytest.mark.asyncio
    async def test_conditional_false_branch_via_legacy_sugar(self):
        ensure_bootstrapped()
        conditional_node = registry.get("conditional_node", namespace="core")
        node_spec = conditional_node(
            "cond_false",
            condition_key="should_continue",
            true_action="retry",
            false_action="proceed",
        )

        result = await node_spec.fn({"should_continue": False})
        assert result["action"] == "proceed"
        assert result["routing"] == "proceed"

    @pytest.mark.asyncio
    async def test_conditional_callable_predicates(self):
        """Enhanced branches: callable predicates."""
        ensure_bootstrapped()
        conditional_node = registry.get("conditional_node", namespace="core")

        node_spec = conditional_node(
            "cond_callable",
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
    async def test_conditional_dotted_path_predicates(self):
        """Enhanced branches: dotted-path predicates."""
        ensure_bootstrapped()
        conditional_node = registry.get("conditional_node", namespace="core")

        node_spec = conditional_node(
            "cond_dotted",
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
    async def test_conditional_else_action(self):
        """Else branch when nothing matches."""
        ensure_bootstrapped()
        conditional_node = registry.get("conditional_node", namespace="core")

        node_spec = conditional_node(
            "cond_else",
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
    async def test_conditional_state_pass_through(self):
        """Predicates can use state passed via ports."""
        ensure_bootstrapped()
        conditional_node = registry.get("conditional_node", namespace="core")

        node_spec = conditional_node(
            "cond_state",
            branches=[
                {"pred": lambda d, s: s.get("flag") is True, "action": "use_state"},
                {"pred": lambda d, s: d.get("v") == 1, "action": "by_data"},
            ],
            else_action="none",
        )

        result = await node_spec.fn({"v": 0}, state={"flag": True})
        assert result["routing"] == "use_state"

    def test_conditional_node_deps_mapping(self):
        """deps passed via kwargs should map to NodeSpec.deps."""
        ensure_bootstrapped()
        conditional_node = registry.get("conditional_node", namespace="core")

        node_spec = conditional_node(
            "cond_with_deps",
            branches=[{"pred": lambda d, s: True, "action": "ok"}],
            deps=["n1", "n2"],
            custom="x",
        )

        assert isinstance(node_spec, NodeSpec)
        assert hasattr(node_spec, "deps")
        assert "n1" in node_spec.deps and "n2" in node_spec.deps
        assert node_spec.params.get("custom") == "x"
