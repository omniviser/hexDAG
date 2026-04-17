"""Tests for pre_seeded_results (pipeline resume)."""

import pytest

from hexdag.kernel.domain.dag import DirectedGraph, NodeSpec
from hexdag.kernel.orchestration.orchestrator import Orchestrator


class TestPreSeededResults:
    @staticmethod
    def _build_linear_graph() -> DirectedGraph:
        """A -> B -> C linear graph."""

        async def node_a(inputs, **kw):
            return {"value": "from_a"}

        async def node_b(inputs, **kw):
            return {"value": f"from_b(got={inputs.get('value', '?')})"}

        async def node_c(inputs, **kw):
            return {"value": f"from_c(got={inputs.get('value', '?')})"}

        graph = DirectedGraph()
        graph += NodeSpec(name="a", fn=node_a)
        graph += NodeSpec(name="b", fn=node_b, deps=frozenset({"a"}))
        graph += NodeSpec(name="c", fn=node_c, deps=frozenset({"b"}))
        return graph

    @pytest.mark.asyncio()
    async def test_no_pre_seeded_runs_all(self):
        graph = self._build_linear_graph()
        orchestrator = Orchestrator()
        results = await orchestrator.run(graph, {})
        assert "a" in results
        assert "b" in results
        assert "c" in results

    @pytest.mark.asyncio()
    async def test_pre_seeded_skips_a(self):
        graph = self._build_linear_graph()
        orchestrator = Orchestrator()
        results = await orchestrator.run(
            graph,
            {},
            pre_seeded_results={"a": {"value": "pre_seeded_a"}},
        )
        # a was pre-seeded — should not have been re-executed
        assert results["a"] == {"value": "pre_seeded_a"}
        # b should have run with a's pre-seeded result
        assert "from_b" in results["b"]["value"]
        # c should have run
        assert "c" in results

    @pytest.mark.asyncio()
    async def test_pre_seeded_skips_a_and_b(self):
        graph = self._build_linear_graph()
        orchestrator = Orchestrator()
        results = await orchestrator.run(
            graph,
            {},
            pre_seeded_results={
                "a": {"value": "seeded_a"},
                "b": {"value": "seeded_b"},
            },
        )
        # a and b were pre-seeded
        assert results["a"] == {"value": "seeded_a"}
        assert results["b"] == {"value": "seeded_b"}
        # c should have run with b's seeded value
        assert "from_c" in results["c"]["value"]

    @pytest.mark.asyncio()
    async def test_pre_seeded_all_returns_immediately(self):
        graph = self._build_linear_graph()
        orchestrator = Orchestrator()
        results = await orchestrator.run(
            graph,
            {},
            pre_seeded_results={
                "a": {"value": "a"},
                "b": {"value": "b"},
                "c": {"value": "c"},
            },
        )
        assert results == {"a": {"value": "a"}, "b": {"value": "b"}, "c": {"value": "c"}}

    @pytest.mark.asyncio()
    async def test_pre_seeded_none_is_noop(self):
        graph = self._build_linear_graph()
        orchestrator = Orchestrator()
        results = await orchestrator.run(graph, {}, pre_seeded_results=None)
        assert "a" in results
        assert "b" in results
        assert "c" in results
