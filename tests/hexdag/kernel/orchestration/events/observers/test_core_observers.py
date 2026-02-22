"""Tests for core observer implementations."""

import asyncio
from unittest.mock import MagicMock

import pytest

from hexdag.kernel.orchestration.events import (
    AlertingObserver,
    DataQualityObserver,
    ExecutionTracerObserver,
    NodeCompleted,
    NodeFailed,
    NodeStarted,
    PerformanceMetricsObserver,
    PipelineCompleted,
    PipelineStarted,
    ResourceMonitorObserver,
    SimpleLoggingObserver,
)

# ==============================================================================
# PERFORMANCE METRICS OBSERVER TESTS
# ==============================================================================


class TestPerformanceMetricsObserver:
    """Test PerformanceMetricsObserver."""

    @pytest.fixture
    def observer(self):
        """Create a PerformanceMetricsObserver instance."""
        return PerformanceMetricsObserver()

    @pytest.mark.asyncio
    async def test_tracks_node_executions(self, observer):
        """Test that observer tracks node execution counts."""
        # Start and complete a node
        await observer.handle(NodeStarted(name="test_node", wave_index=0, dependencies=[]))
        await observer.handle(
            NodeCompleted(name="test_node", wave_index=0, result={"data": 123}, duration_ms=100.0)
        )

        summary = observer.get_summary()
        assert summary["total_nodes_executed"] == 1
        assert summary["node_executions"]["test_node"] == 1

    @pytest.mark.asyncio
    async def test_tracks_node_timings(self, observer):
        """Test that observer tracks node execution times."""
        await observer.handle(NodeStarted(name="node1", wave_index=0, dependencies=[]))
        await observer.handle(
            NodeCompleted(name="node1", wave_index=0, result={}, duration_ms=150.0)
        )

        summary = observer.get_summary()
        assert "node1" in summary["average_timings_ms"]
        assert summary["average_timings_ms"]["node1"] == 150.0
        assert summary["min_timings_ms"]["node1"] == 150.0
        assert summary["max_timings_ms"]["node1"] == 150.0

    @pytest.mark.asyncio
    async def test_tracks_multiple_executions(self, observer):
        """Test tracking multiple executions of same node."""
        # Execute same node 3 times with different durations
        for duration in [100.0, 200.0, 150.0]:
            await observer.handle(NodeStarted(name="node1", wave_index=0, dependencies=[]))
            await observer.handle(
                NodeCompleted(name="node1", wave_index=0, result={}, duration_ms=duration)
            )

        summary = observer.get_summary()
        assert summary["node_executions"]["node1"] == 3
        assert summary["average_timings_ms"]["node1"] == 150.0  # (100+200+150)/3
        assert summary["min_timings_ms"]["node1"] == 100.0
        assert summary["max_timings_ms"]["node1"] == 200.0

    @pytest.mark.asyncio
    async def test_tracks_failures(self, observer):
        """Test that observer tracks failures."""
        await observer.handle(NodeStarted(name="failing_node", wave_index=0, dependencies=[]))
        await observer.handle(
            NodeFailed(name="failing_node", wave_index=0, error=Exception("Test error"))
        )

        summary = observer.get_summary()
        assert summary["failures"]["failing_node"] == 1
        assert summary["total_failures"] == 1

    @pytest.mark.asyncio
    async def test_calculates_success_rate(self, observer):
        """Test success rate calculation."""
        # 2 successes
        for _ in range(2):
            await observer.handle(NodeStarted(name="node1", wave_index=0, dependencies=[]))
            await observer.handle(
                NodeCompleted(name="node1", wave_index=0, result={}, duration_ms=100.0)
            )

        # 1 failure
        await observer.handle(NodeStarted(name="node2", wave_index=0, dependencies=[]))
        await observer.handle(NodeFailed(name="node2", wave_index=0, error=Exception("Error")))

        summary = observer.get_summary()
        assert summary["total_nodes_executed"] == 3
        assert summary["overall_success_rate"] == pytest.approx(66.67, rel=0.1)

    @pytest.mark.asyncio
    async def test_tracks_pipeline_times(self, observer):
        """Test tracking pipeline start and end times."""
        await observer.handle(PipelineStarted(name="pipeline1", total_waves=5, total_nodes=10))
        await observer.handle(PipelineCompleted(name="pipeline1", duration_ms=1000.0))

        assert "pipeline1" in observer.pipeline_start_times
        assert "pipeline1" in observer.pipeline_end_times

    @pytest.mark.asyncio
    async def test_reset(self, observer):
        """Test reset functionality."""
        # Add some data
        await observer.handle(NodeStarted(name="node1", wave_index=0, dependencies=[]))
        await observer.handle(
            NodeCompleted(name="node1", wave_index=0, result={}, duration_ms=100.0)
        )

        observer.reset()

        summary = observer.get_summary()
        assert summary["total_nodes_executed"] == 0
        assert len(summary["node_executions"]) == 0


# ==============================================================================
# ALERTING OBSERVER TESTS
# ==============================================================================


class TestAlertingObserver:
    """Test AlertingObserver."""

    @pytest.fixture
    def observer(self):
        """Create an AlertingObserver instance."""
        return AlertingObserver(slow_threshold_ms=500.0)

    @pytest.mark.asyncio
    async def test_triggers_slow_node_alert(self, observer):
        """Test that slow nodes trigger alerts."""
        await observer.handle(
            NodeCompleted(name="slow_node", wave_index=0, result={}, duration_ms=1000.0)
        )

        alerts = observer.get_alerts()
        assert len(alerts) == 1
        assert alerts[0].type.value == "SLOW_NODE"
        assert alerts[0].node == "slow_node"
        assert alerts[0].duration_ms == 1000.0

    @pytest.mark.asyncio
    async def test_no_alert_for_fast_nodes(self, observer):
        """Test that fast nodes don't trigger alerts."""
        await observer.handle(
            NodeCompleted(name="fast_node", wave_index=0, result={}, duration_ms=100.0)
        )

        alerts = observer.get_alerts()
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_triggers_failure_alert(self, observer):
        """Test that failures trigger alerts."""
        error = Exception("Test failure")
        await observer.handle(NodeFailed(name="failed_node", wave_index=0, error=error))

        alerts = observer.get_alerts()
        assert len(alerts) == 1
        assert alerts[0].type.value == "NODE_FAILURE"
        assert alerts[0].node == "failed_node"

    @pytest.mark.asyncio
    async def test_calls_callback_on_alert(self, observer):
        """Test that callback is called when alert is triggered."""
        callback = MagicMock()
        observer.on_alert = callback

        await observer.handle(
            NodeCompleted(name="slow_node", wave_index=0, result={}, duration_ms=1000.0)
        )

        callback.assert_called_once()
        alert = callback.call_args[0][0]
        assert alert.type.value == "SLOW_NODE"

    @pytest.mark.asyncio
    async def test_get_alerts_by_type(self, observer):
        """Test filtering alerts by type."""
        # Trigger both types of alerts
        await observer.handle(
            NodeCompleted(name="slow_node", wave_index=0, result={}, duration_ms=1000.0)
        )
        await observer.handle(NodeFailed(name="failed_node", wave_index=0, error=Exception("err")))

        slow_alerts = observer.get_alerts(alert_type="SLOW_NODE")
        fail_alerts = observer.get_alerts(alert_type="NODE_FAILURE")

        assert len(slow_alerts) == 1
        assert len(fail_alerts) == 1

    @pytest.mark.asyncio
    async def test_clear_alerts(self, observer):
        """Test clearing alerts."""
        await observer.handle(
            NodeCompleted(name="slow_node", wave_index=0, result={}, duration_ms=1000.0)
        )

        observer.clear_alerts()
        assert len(observer.get_alerts()) == 0


# ==============================================================================
# SIMPLE LOGGING OBSERVER TESTS
# ==============================================================================


class TestSimpleLoggingObserver:
    """Test SimpleLoggingObserver."""

    @pytest.fixture
    def observer(self):
        """Create a SimpleLoggingObserver instance."""
        return SimpleLoggingObserver(verbose=False)

    @pytest.mark.asyncio
    async def test_handles_node_started(self, observer):
        """Test handling NodeStarted events."""
        # Should not raise
        await observer.handle(NodeStarted(name="node1", wave_index=0, dependencies=[]))

    @pytest.mark.asyncio
    async def test_handles_node_completed(self, observer):
        """Test handling NodeCompleted events."""
        await observer.handle(
            NodeCompleted(name="node1", wave_index=0, result={"data": 123}, duration_ms=100.0)
        )

    @pytest.mark.asyncio
    async def test_handles_node_failed(self, observer):
        """Test handling NodeFailed events."""
        await observer.handle(NodeFailed(name="node1", wave_index=0, error=Exception("error")))

    @pytest.mark.asyncio
    async def test_handles_pipeline_events(self, observer):
        """Test handling pipeline events."""
        await observer.handle(PipelineStarted(name="pipeline", total_waves=5, total_nodes=10))
        await observer.handle(PipelineCompleted(name="pipeline", duration_ms=1000.0))


# ==============================================================================
# RESOURCE MONITOR OBSERVER TESTS
# ==============================================================================


class TestResourceMonitorObserver:
    """Test ResourceMonitorObserver."""

    @pytest.fixture
    def observer(self):
        """Create a ResourceMonitorObserver instance."""
        return ResourceMonitorObserver()

    @pytest.mark.asyncio
    async def test_tracks_concurrent_nodes(self, observer):
        """Test tracking concurrent node execution."""
        # Start 3 nodes concurrently
        await observer.handle(NodeStarted(name="node1", wave_index=0, dependencies=[]))
        await observer.handle(NodeStarted(name="node2", wave_index=0, dependencies=[]))
        await observer.handle(NodeStarted(name="node3", wave_index=0, dependencies=[]))

        stats = observer.get_stats()
        assert stats["max_concurrent"] == 3

    @pytest.mark.asyncio
    async def test_tracks_wave_sizes(self, observer):
        """Test tracking wave sizes."""
        # Wave 0: 2 nodes
        await observer.handle(NodeStarted(name="node1", wave_index=0, dependencies=[]))
        await observer.handle(NodeStarted(name="node2", wave_index=0, dependencies=[]))
        await observer.handle(
            NodeCompleted(name="node1", wave_index=0, result={}, duration_ms=100.0)
        )
        await observer.handle(
            NodeCompleted(name="node2", wave_index=0, result={}, duration_ms=100.0)
        )

        # Wave 1: 1 node
        await observer.handle(NodeStarted(name="node3", wave_index=1, dependencies=["node1"]))
        await observer.handle(
            NodeCompleted(name="node3", wave_index=1, result={}, duration_ms=100.0)
        )

        stats = observer.get_stats()
        assert stats["total_waves"] >= 1

    @pytest.mark.asyncio
    async def test_decrements_concurrent_on_completion(self, observer):
        """Test that concurrent count decrements on completion."""
        await observer.handle(NodeStarted(name="node1", wave_index=0, dependencies=[]))
        await observer.handle(NodeStarted(name="node2", wave_index=0, dependencies=[]))

        # Complete one node
        await observer.handle(
            NodeCompleted(name="node1", wave_index=0, result={}, duration_ms=100.0)
        )

        assert observer.concurrent_nodes == 1

    @pytest.mark.asyncio
    async def test_reset(self, observer):
        """Test reset functionality."""
        await observer.handle(NodeStarted(name="node1", wave_index=0, dependencies=[]))

        observer.reset()
        assert observer.max_concurrent == 0
        assert observer.concurrent_nodes == 0


# ==============================================================================
# DATA QUALITY OBSERVER TESTS
# ==============================================================================


class TestDataQualityObserver:
    """Test DataQualityObserver."""

    @pytest.fixture
    def observer(self):
        """Create a DataQualityObserver instance."""
        return DataQualityObserver()

    @pytest.mark.asyncio
    async def test_detects_none_results(self, observer):
        """Test detection of None results."""
        await observer.handle(
            NodeCompleted(name="node1", wave_index=0, result=None, duration_ms=100.0)
        )

        assert observer.has_issues()
        issues = observer.get_issues()
        assert len(issues) == 1
        assert issues[0].metadata["issue_type"] == "null_result"

    @pytest.mark.asyncio
    async def test_detects_empty_list(self, observer):
        """Test detection of empty list results."""
        await observer.handle(
            NodeCompleted(name="node1", wave_index=0, result=[], duration_ms=100.0)
        )

        issues = observer.get_issues()
        assert len(issues) == 1
        assert issues[0].metadata["issue_type"] == "empty_result"

    @pytest.mark.asyncio
    async def test_detects_empty_dict(self, observer):
        """Test detection of empty dict results."""
        await observer.handle(
            NodeCompleted(name="node1", wave_index=0, result={}, duration_ms=100.0)
        )

        issues = observer.get_issues()
        assert len(issues) == 1
        assert issues[0].metadata["issue_type"] == "empty_result"

    @pytest.mark.asyncio
    async def test_detects_error_flag(self, observer):
        """Test detection of error flags in results."""
        await observer.handle(
            NodeCompleted(
                name="node1",
                wave_index=0,
                result={"error": "Something went wrong"},
                duration_ms=100.0,
            )
        )

        issues = observer.get_issues()
        assert any(i.metadata.get("issue_type") == "error_in_result" for i in issues)

    @pytest.mark.asyncio
    async def test_detects_error_status(self, observer):
        """Test detection of error status codes."""
        await observer.handle(
            NodeCompleted(
                name="node1", wave_index=0, result={"status": "failed"}, duration_ms=100.0
            )
        )

        issues = observer.get_issues()
        assert any(i.metadata.get("issue_type") == "error_status" for i in issues)

    @pytest.mark.asyncio
    async def test_no_issues_for_valid_data(self, observer):
        """Test that valid data doesn't trigger issues."""
        await observer.handle(
            NodeCompleted(name="node1", wave_index=0, result={"data": [1, 2, 3]}, duration_ms=100.0)
        )

        assert not observer.has_issues()

    @pytest.mark.asyncio
    async def test_filter_by_severity(self, observer):
        """Test filtering issues by severity."""
        # Trigger warning
        await observer.handle(
            NodeCompleted(name="node1", wave_index=0, result=None, duration_ms=100.0)
        )

        # Trigger error
        await observer.handle(
            NodeCompleted(name="node2", wave_index=0, result={"error": "failed"}, duration_ms=100.0)
        )

        warnings = observer.get_issues(severity="warning")
        errors = observer.get_issues(severity="error")

        assert len(warnings) >= 1
        assert len(errors) >= 1

    @pytest.mark.asyncio
    async def test_clear_issues(self, observer):
        """Test clearing issues."""
        await observer.handle(
            NodeCompleted(name="node1", wave_index=0, result=None, duration_ms=100.0)
        )

        observer.clear_issues()
        assert not observer.has_issues()
        assert observer.validated_nodes == 0


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================


class TestObserverIntegration:
    """Integration tests for observers working together."""

    @pytest.mark.asyncio
    async def test_multiple_observers_same_event(self):
        """Test multiple observers can handle the same event."""
        metrics = PerformanceMetricsObserver()
        tracer = ExecutionTracerObserver()
        quality = DataQualityObserver()

        # Send both start and complete events
        start_event = NodeStarted(name="node1", wave_index=0, dependencies=[])
        complete_event = NodeCompleted(
            name="node1", wave_index=0, result={"data": 123}, duration_ms=100.0
        )

        # All observers handle both events
        for event in [start_event, complete_event]:
            await asyncio.gather(
                metrics.handle(event),
                tracer.handle(event),
                quality.handle(event),
            )

        # Each observer should have processed them
        assert metrics.get_summary()["total_nodes_executed"] == 1
        assert len(tracer.get_trace().events) == 2
        assert not quality.has_issues()

    @pytest.mark.asyncio
    async def test_observer_isolation(self):
        """Test that observer failures don't affect each other."""

        class FailingObserver:
            async def handle(self, event):
                raise Exception("Observer error")

        failing = FailingObserver()
        metrics = PerformanceMetricsObserver()

        event = NodeStarted(name="node1", wave_index=0, dependencies=[])

        # Even if one fails, others should work
        try:
            await failing.handle(event)
        except Exception:
            pass

        await metrics.handle(event)
        assert metrics.get_summary()["total_nodes_executed"] == 1
