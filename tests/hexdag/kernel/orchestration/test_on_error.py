"""Tests for on_error handler routing in the orchestrator."""

import pytest

from hexdag.kernel.domain.dag import DirectedGraph, NodeSpec
from hexdag.kernel.exceptions import OrchestratorError
from hexdag.kernel.orchestration.hook_context import PipelineStatus
from hexdag.kernel.orchestration.orchestrator import Orchestrator
from hexdag.kernel.ports.executor import ExecutionResult


def _dummy_fn(**kwargs):
    return "ok"


def _make_result(
    name: str,
    *,
    status: PipelineStatus = PipelineStatus.SUCCESS,
    output: object = "ok",
    error: str | None = None,
    error_type: str | None = None,
) -> ExecutionResult:
    return ExecutionResult(
        node_name=name,
        output=output,
        duration_ms=1.0,
        status=status,
        error=error,
        error_type=error_type,
    )


class TestCheckWaveResultsForFailures:
    """Unit tests for Orchestrator._check_wave_results_for_failures."""

    def _make_orchestrator(self) -> Orchestrator:
        return Orchestrator()

    def test_all_success_passes(self):
        """All-success wave raises nothing."""
        orch = self._make_orchestrator()
        results = {"a": _make_result("a")}
        # Should not raise
        orch._check_wave_results_for_failures(results)

    def test_unhandled_failure_raises(self):
        """A failed node without on_error raises OrchestratorError."""
        orch = self._make_orchestrator()
        results = {
            "a": _make_result(
                "a",
                status=PipelineStatus.FAILED,
                error="boom",
                error_type="ValueError",
            ),
        }
        with pytest.raises(OrchestratorError, match="boom"):
            orch._check_wave_results_for_failures(results)

    def test_handled_failure_injects_error_payload(self):
        """A failed node WITH on_error injects _error payload into node_results."""
        orch = self._make_orchestrator()
        graph = DirectedGraph()
        graph.add(NodeSpec("risky", _dummy_fn, on_error="handler"))
        graph.add(NodeSpec("handler", _dummy_fn).after("risky"))

        node_results: dict = {}
        results = {
            "risky": _make_result(
                "risky",
                status=PipelineStatus.FAILED,
                error="connection timeout",
                error_type="TimeoutError",
            ),
        }
        # Should NOT raise
        orch._check_wave_results_for_failures(results, graph, node_results)

        assert "_error" in node_results["risky"]
        assert node_results["risky"]["_error"]["node"] == "risky"
        assert node_results["risky"]["_error"]["type"] == "TimeoutError"
        assert node_results["risky"]["_error"]["message"] == "connection timeout"

    def test_mixed_handled_and_unhandled(self):
        """Only unhandled failures raise; handled ones inject payload."""
        orch = self._make_orchestrator()
        graph = DirectedGraph()
        graph.add(NodeSpec("handled_node", _dummy_fn, on_error="h"))
        graph.add(NodeSpec("h", _dummy_fn).after("handled_node"))
        graph.add(NodeSpec("unhandled_node", _dummy_fn))

        node_results: dict = {}
        results = {
            "handled_node": _make_result(
                "handled_node",
                status=PipelineStatus.FAILED,
                error="e1",
                error_type="E1",
            ),
            "unhandled_node": _make_result(
                "unhandled_node",
                status=PipelineStatus.FAILED,
                error="e2",
                error_type="E2",
            ),
        }
        with pytest.raises(OrchestratorError, match="unhandled_node"):
            orch._check_wave_results_for_failures(results, graph, node_results)

        # handled_node's payload was still injected before the raise
        assert "_error" in node_results["handled_node"]

    def test_no_graph_treats_all_as_unhandled(self):
        """When graph is None, all failures are unhandled."""
        orch = self._make_orchestrator()
        results = {
            "a": _make_result(
                "a",
                status=PipelineStatus.FAILED,
                error="err",
                error_type="X",
            ),
        }
        with pytest.raises(OrchestratorError):
            orch._check_wave_results_for_failures(results, graph=None, node_results={})

    def test_default_error_values(self):
        """Missing error/error_type use fallback strings."""
        orch = self._make_orchestrator()
        graph = DirectedGraph()
        graph.add(NodeSpec("risky", _dummy_fn, on_error="handler"))
        graph.add(NodeSpec("handler", _dummy_fn).after("risky"))

        node_results: dict = {}
        results = {
            "risky": _make_result(
                "risky",
                status=PipelineStatus.FAILED,
                error=None,
                error_type=None,
            ),
        }
        orch._check_wave_results_for_failures(results, graph, node_results)

        assert node_results["risky"]["_error"]["type"] == "Unknown"
        assert node_results["risky"]["_error"]["message"] == "No error message"

    def test_multiple_nodes_same_handler(self):
        """Multiple failed nodes can route to the same handler."""
        orch = self._make_orchestrator()
        graph = DirectedGraph()
        graph.add(NodeSpec("a", _dummy_fn, on_error="handler"))
        graph.add(NodeSpec("b", _dummy_fn, on_error="handler"))
        graph.add(NodeSpec("handler", _dummy_fn).after("a", "b"))

        node_results: dict = {}
        results = {
            "a": _make_result("a", status=PipelineStatus.FAILED, error="err_a", error_type="A"),
            "b": _make_result("b", status=PipelineStatus.FAILED, error="err_b", error_type="B"),
        }
        # Should NOT raise — both failures are handled
        orch._check_wave_results_for_failures(results, graph, node_results)

        assert node_results["a"]["_error"]["message"] == "err_a"
        assert node_results["b"]["_error"]["message"] == "err_b"
