"""Tests for LoopNode and ConditionalNode (builder-first, using only registry objects)."""

import asyncio

import pytest

from hexdag.builtin.nodes.loop_node import StopReason
from hexdag.core.bootstrap import ensure_bootstrapped
from hexdag.core.domain.dag import NodeSpec
from hexdag.core.registry import registry

# Ensure registry is bootstrapped for tests
ensure_bootstrapped()


class TestLoopNode:
    """LoopNode via fluent builder, using only the object fetched from registry."""

    @pytest.mark.asyncio
    async def test_collect_last(self):
        loop_node = registry.get("loop_node", namespace="core")

        i_key = "i"

        def process(data, state):
            return state.get(i_key, 0)

        def on_end(state, out):
            return {i_key: state.get(i_key, 0) + 1}

        spec: NodeSpec = (
            loop_node.name("retry")
            .condition(lambda d, s: s.get(i_key, 0) < 3)
            .do(process)
            .on_iteration_end(on_end)
            .init_state({i_key: 0})
            .collect_last()
            .max_iterations(10)
            .build()
        )
        assert isinstance(spec, NodeSpec)

        result = await spec.fn({"any": "payload"})
        assert result["result"] == 2
        assert result["metadata"]["iterations"] == 3
        assert result["metadata"]["state"][i_key] == 3
        stopped = result["metadata"]["stopped_by"]
        stopped = stopped.value if hasattr(stopped, "value") else stopped
        assert stopped == StopReason.CONDITION.value

    @pytest.mark.asyncio
    async def test_collect_list(self):
        loop_node = registry.get("loop_node", namespace="core")

        def body(data, state):
            return f"v-{state.get('k', 0)}"

        def on_end(state, out):
            return {"k": state.get("k", 0) + 1}

        spec = (
            loop_node.name("listy")
            .condition(lambda d, s: s.get("k", 0) < 4)
            .do(body)
            .on_iteration_end(on_end)
            .init_state({"k": 0})
            .collect_list()
            .max_iterations(10)
            .build()
        )
        assert isinstance(spec, NodeSpec)

        result = await spec.fn({"x": 1})
        assert result["result"] == ["v-0", "v-1", "v-2", "v-3"]
        assert result["metadata"]["iterations"] == 4
        assert result["metadata"]["state"]["k"] == 4

    @pytest.mark.asyncio
    async def test_collect_reduce(self):
        loop_node = registry.get("loop_node", namespace="core")

        def body(data, state):
            return state.get("n", 0)

        def on_end(state, out):
            return {"n": state.get("n", 0) + 1}

        def reducer(acc, x):
            return (acc or 0) + (x or 0)

        spec = (
            loop_node.name("reducer")
            .condition(lambda d, s: s.get("n", 0) < 5)
            .do(body)
            .on_iteration_end(on_end)
            .init_state({"n": 0})
            .collect_reduce(reducer)
            .max_iterations(10)
            .build()
        )
        assert isinstance(spec, NodeSpec)

        result = await spec.fn({"input": True})
        assert result["result"] == 10
        assert result["metadata"]["iterations"] == 5
        assert result["metadata"]["state"]["n"] == 5
        stopped = result["metadata"]["stopped_by"]
        stopped = stopped.value if hasattr(stopped, "value") else stopped
        assert stopped == StopReason.CONDITION.value

    @pytest.mark.asyncio
    async def test_loop_node_custom_iteration_key(self):
        """Test loop node with custom iteration key."""
        ensure_bootstrapped()
        loop_node = registry.get("loop_node", namespace="core")

        node_spec = (
            loop_node.name("test_loop")
            .condition(lambda d, s: s.get("custom_iteration", 0) < 2)
            .do(lambda d, s: s.get("custom_iteration", 0))
            .on_iteration_end(lambda s, o: {"custom_iteration": s.get("custom_iteration", 0) + 1})
            .init_state({"custom_iteration": 0})
            .iteration_key("custom_iteration")
            .max_iterations(2)
            .build()
        )

        # Verify node was created with custom iteration key
        assert node_spec.name == "test_loop"
        assert callable(node_spec.fn)

        # Test execution to verify custom key works
        result = await node_spec.fn({"input": "test"})
        assert result["metadata"]["state"]["custom_iteration"] == 2

    def test_loop_node_invalid_max_iterations(self):
        """Test loop node with invalid max_iterations."""
        ensure_bootstrapped()
        loop_node = registry.get("loop_node", namespace="core")

        # Test with max_iterations = 0
        with pytest.raises(ValueError, match="max_iterations must be positive"):
            (loop_node.name("test_loop").condition(lambda d, s: True).max_iterations(0).build())

        # Test with max_iterations = -1
        with pytest.raises(ValueError, match="max_iterations must be positive"):
            (loop_node.name("test_loop").condition(lambda d, s: True).max_iterations(-1).build())

    async def test_break_if(self):
        loop_node = registry.get("loop_node", namespace="core")

        def body(data, state):
            return state.get("i", 0)

        def on_end(state, out):
            return {"i": state.get("i", 0) + 1}

        spec = (
            loop_node.name("breaker")
            .condition(lambda d, s: True)
            .do(body)
            .on_iteration_end(on_end)
            .init_state({"i": 0})
            .collect_list()
            .break_if(lambda d, s: s.get("i", 0) >= 3)
            .iteration_key("iter")
            .max_iterations(10)
            .build()
        )
        assert isinstance(spec, NodeSpec)

        result = await spec.fn({"flags": {"skip": False}})
        assert result["result"] == [0, 1, 2]
        assert result["metadata"]["iterations"] == 3
        assert result["metadata"]["state"]["i"] == 3
        stopped = result["metadata"]["stopped_by"]
        stopped = stopped.value if hasattr(stopped, "value") else stopped
        assert stopped == StopReason.BREAK_GUARD.value

    @pytest.mark.asyncio
    async def test_max_iterations_safety(self):
        loop_node = registry.get("loop_node", namespace="core")

        def body(data, state):
            return "x"

        def on_end(state, out):
            return {"c": state.get("c", 0) + 1}

        spec = (
            loop_node.name("safety")
            .condition(lambda d, s: True)
            .do(body)
            .on_iteration_end(on_end)
            .init_state({"c": 0})
            .collect_list()
            .max_iterations(2)
            .build()
        )
        assert isinstance(spec, NodeSpec)

        result = await spec.fn({"k": "v"})
        assert result["result"] == ["x", "x"]
        assert result["metadata"]["iterations"] == 2
        stopped = result["metadata"]["stopped_by"]
        stopped = stopped.value if hasattr(stopped, "value") else stopped
        assert stopped == StopReason.LIMIT.value


class TestConditionalNode:
    """ConditionalNode via fluent builder, using only the object fetched from registry."""

    def test_basic(self):
        conditional_node = registry.get("conditional_node", namespace="core")

        spec: NodeSpec = (
            conditional_node.name("router")
            .when(lambda d, s: d.get("kind") == "A", "ACTION_A")
            .when(lambda d, s: d.get("kind") == "B", "ACTION_B")
            .otherwise("ACTION_DEFAULT")
            .build()
        )
        assert isinstance(spec, NodeSpec)

        result = asyncio.get_event_loop().run_until_complete(spec.fn({"kind": "B"}))
        assert result["result"] == "ACTION_B"
        assert result["metadata"]["evaluations"] == [False, True]
        assert result["metadata"]["has_else"] is True

    def test_deps_and_models_passthrough(self):
        conditional_node = registry.get("conditional_node", namespace="core")

        class InModel:
            pass

        class OutModel:
            pass

        spec: NodeSpec = (
            conditional_node.name("router2")
            .when(lambda d, s: True, "OK")
            .deps({"dep1", "dep2"})
            .in_model(InModel)
            .out_model(OutModel)
            .build()
        )
        assert isinstance(spec, NodeSpec)
        assert "dep1" in spec.deps and "dep2" in spec.deps
        assert spec.in_model is InModel
        assert spec.out_model is OutModel

    def test_otherwise_only(self):
        conditional_node = registry.get("conditional_node", namespace="core")

        spec: NodeSpec = conditional_node.name("fallback_only").otherwise("DEFAULT").build()
        assert isinstance(spec, NodeSpec)

        result = asyncio.get_event_loop().run_until_complete(spec.fn({"x": 1}))
        assert result["result"] == "DEFAULT"
        assert result["metadata"]["has_else"] is True
