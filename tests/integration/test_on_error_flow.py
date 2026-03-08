"""Integration tests for on_error handler end-to-end flow."""

import pytest

from hexdag.kernel.domain.dag import DirectedGraph, NodeSpec
from hexdag.kernel.exceptions import OrchestratorError
from hexdag.kernel.orchestration.orchestrator import Orchestrator


async def _failing_node(x=None, **kwargs):
    """Node that always raises."""
    raise ValueError("intentional failure")


async def _error_handler(inputs=None, **kwargs):
    """Handler that recovers from an upstream failure."""
    if isinstance(inputs, dict) and "_error" in inputs:
        return {"handled": True, "original_error": inputs["_error"]["message"]}
    return {"handled": False, "raw_input": inputs}


async def _ok_node(x=None, **kwargs):
    """A node that succeeds."""
    return {"status": "ok"}


async def _bad_handler(inputs=None, **kwargs):
    """Handler that itself fails."""
    raise RuntimeError("handler also failed")


class TestOnErrorIntegration:
    """End-to-end tests for on_error handler pipeline execution."""

    @pytest.mark.asyncio
    async def test_handler_receives_error_payload(self):
        """Full pipeline: failed node -> handler receives _error payload."""
        graph = DirectedGraph()
        graph.add(NodeSpec("risky", _failing_node, on_error="handler"))
        graph.add(NodeSpec("handler", _error_handler).after("risky"))

        orchestrator = Orchestrator()
        results = await orchestrator.run(graph, {"data": "test"})

        assert results["handler"]["handled"] is True
        assert "intentional failure" in results["handler"]["original_error"]

    @pytest.mark.asyncio
    async def test_handler_failure_propagates(self):
        """If the handler itself fails, OrchestratorError is raised."""
        graph = DirectedGraph()
        graph.add(NodeSpec("risky", _failing_node, on_error="bad_handler"))
        graph.add(NodeSpec("bad_handler", _bad_handler).after("risky"))

        orchestrator = Orchestrator()
        with pytest.raises(OrchestratorError, match="bad_handler"):
            await orchestrator.run(graph, {})

    @pytest.mark.asyncio
    async def test_success_skips_handler_path(self):
        """When a node succeeds, the handler still runs but gets normal output."""
        graph = DirectedGraph()
        graph.add(NodeSpec("maybe_risky", _ok_node, on_error="handler"))
        graph.add(NodeSpec("handler", _error_handler).after("maybe_risky"))

        orchestrator = Orchestrator()
        results = await orchestrator.run(graph, {})

        # Node succeeded, handler runs with normal output (no _error key)
        assert "maybe_risky" in results
        assert results["maybe_risky"]["status"] == "ok"

    @pytest.mark.asyncio
    async def test_chained_error_handler(self):
        """Handler can itself have an on_error for cascading recovery."""
        graph = DirectedGraph()
        graph.add(NodeSpec("risky", _failing_node, on_error="handler_1"))
        graph.add(NodeSpec("handler_1", _bad_handler, on_error="handler_2").after("risky"))
        graph.add(NodeSpec("handler_2", _error_handler).after("handler_1"))

        orchestrator = Orchestrator()
        results = await orchestrator.run(graph, {})

        # handler_1 failed, so handler_2 gets the _error from handler_1
        assert results["handler_2"]["handled"] is True
        assert "handler also failed" in results["handler_2"]["original_error"]
