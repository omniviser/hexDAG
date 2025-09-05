"""Tests for observer implementations."""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hexai.core.application.events import (
    FileObserver,
    LoggingObserver,
    MetricsObserver,
    NodeCompleted,
    NodeFailed,
    NodeStarted,
    NodeStateObserver,
    ObserverManager,
    PipelineCompleted,
    PipelineStarted,
    WebSocketObserver,
)


class TestLoggingObserver:
    """Test the logging observer."""

    @pytest.mark.asyncio
    async def test_logging_observer_handles_events(self):
        """Test that LoggingObserver logs events correctly."""
        observer = LoggingObserver()

        # Mock the logger
        with patch.object(observer.logger, "log") as mock_log:
            # Handle a pipeline started event
            event = PipelineStarted(name="test_pipeline", total_nodes=5, total_waves=2)
            await observer.handle(event)

            # Give time for async execution
            await asyncio.sleep(0.1)

            # Logger should have been called
            mock_log.assert_called()

    @pytest.mark.asyncio
    async def test_logging_observer_error_handling(self):
        """Test that LoggingObserver handles errors in nodes."""
        observer = LoggingObserver()

        # Mock the logger
        with patch.object(observer.logger, "error") as mock_error:
            # Handle a node failed event
            error = RuntimeError("Test error")
            event = NodeFailed(name="failing_node", wave_index=1, error=error)
            await observer.handle(event)

            # Give time for async execution
            await asyncio.sleep(0.1)

            # Error logger should have been called
            mock_error.assert_called()


class TestMetricsObserver:
    """Test the metrics observer."""

    @pytest.mark.asyncio
    async def test_metrics_observer_collects_stats(self):
        """Test that MetricsObserver collects execution metrics."""
        observer = MetricsObserver()

        # Simulate a pipeline execution
        await observer.handle(PipelineStarted(name="test", total_nodes=2, total_waves=1))
        await observer.handle(NodeStarted(name="node1", wave_index=1))
        await observer.handle(NodeCompleted(name="node1", wave_index=1, result={}, duration_ms=100))
        await observer.handle(NodeStarted(name="node2", wave_index=1))
        await observer.handle(NodeCompleted(name="node2", wave_index=1, result={}, duration_ms=200))
        await observer.handle(PipelineCompleted(name="test", duration_ms=300))

        # Check metrics
        summary = observer.get_summary()
        assert summary["total_nodes"] == 2
        assert summary["completed_nodes"] == 2
        assert summary["failed_nodes"] == 0
        assert abs(summary["average_node_time"] - 0.15) < 0.001

    @pytest.mark.asyncio
    async def test_metrics_observer_tracks_failures(self):
        """Test that MetricsObserver tracks failed nodes."""
        observer = MetricsObserver()

        await observer.handle(PipelineStarted(name="test", total_nodes=3, total_waves=1))
        await observer.handle(NodeStarted(name="node1", wave_index=1))
        await observer.handle(NodeCompleted(name="node1", wave_index=1, result={}, duration_ms=100))
        await observer.handle(NodeStarted(name="node2", wave_index=1))
        await observer.handle(NodeFailed(name="node2", wave_index=1, error=RuntimeError("test")))

        summary = observer.get_summary()
        assert summary["completed_nodes"] == 1
        assert summary["failed_nodes"] == 1

    def test_metrics_observer_reset(self):
        """Test that MetricsObserver can be reset."""
        observer = MetricsObserver()

        # Add some data
        observer.completed_nodes = 5
        observer.failed_nodes = 2

        # Reset
        observer.reset()

        # Should be cleared
        assert observer.completed_nodes == 0
        assert observer.failed_nodes == 0
        assert observer.total_nodes == 0


class TestFileObserver:
    """Test the file observer."""

    @pytest.mark.asyncio
    async def test_file_observer_writes_events(self):
        """Test that FileObserver writes events to a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "events.jsonl"
            observer = FileObserver(file_path)

            # Write some events
            events = [
                NodeStarted(name="node1", wave_index=1),
                NodeCompleted(
                    name="node1", wave_index=1, result={"output": "test"}, duration_ms=100
                ),
            ]

            for event in events:
                await observer.handle(event)

            # Give time for async writes
            await asyncio.sleep(0.1)

            # Check file was created and has content
            assert file_path.exists()

            # Read and verify content
            with open(file_path) as f:
                lines = f.readlines()
                assert len(lines) == 2

                # Each line should be valid JSON
                for line in lines:
                    data = json.loads(line)
                    assert "type" in data
                    assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_file_observer_creates_directory(self):
        """Test that FileObserver creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Nested path that doesn't exist
            file_path = Path(tmpdir) / "logs" / "events" / "test.jsonl"
            observer = FileObserver(file_path)

            # Parent directory should be created
            assert file_path.parent.exists()

            # Write an event
            await observer.handle(NodeStarted(name="test", wave_index=1))
            await asyncio.sleep(0.1)

            # File should exist
            assert file_path.exists()


class TestNodeStateObserver:
    """Test the node state observer."""

    @pytest.mark.asyncio
    async def test_node_state_observer_tracks_state(self):
        """Test that NodeStateObserver tracks node execution state."""
        observer = NodeStateObserver()

        # Start a node
        await observer.handle(NodeStarted(name="test_node", wave_index=1, dependencies=["dep1"]))

        # Check state
        state = observer.get_node_state("test_node")
        assert state is not None
        assert state["status"] == "running"
        assert state["wave_index"] == 1
        assert state["dependencies"] == ["dep1"]

        # Complete the node
        await observer.handle(
            NodeCompleted(
                name="test_node", wave_index=1, result={"output": "test"}, duration_ms=500
            )
        )

        # Check updated state
        state = observer.get_node_state("test_node")
        assert state["status"] == "completed"
        assert state["duration_ms"] == 500
        assert state["result"] == {"output": "test"}

    @pytest.mark.asyncio
    async def test_node_state_observer_tracks_failures(self):
        """Test that NodeStateObserver tracks failed nodes."""
        observer = NodeStateObserver()

        # Start and fail a node
        await observer.handle(NodeStarted(name="failing_node", wave_index=1))
        await observer.handle(
            NodeFailed(name="failing_node", wave_index=1, error=RuntimeError("Test error"))
        )

        # Check state
        state = observer.get_node_state("failing_node")
        assert state["status"] == "failed"
        assert "Test error" in state["error"]

    def test_node_state_observer_running_nodes(self):
        """Test getting list of running nodes."""
        observer = NodeStateObserver()

        # Set up some node states
        observer.node_states = {
            "node1": {"status": "running"},
            "node2": {"status": "completed"},
            "node3": {"status": "running"},
            "node4": {"status": "failed"},
        }

        running = observer.get_running_nodes()
        assert set(running) == {"node1", "node3"}

    def test_node_state_observer_failed_nodes(self):
        """Test getting list of failed nodes."""
        observer = NodeStateObserver()

        # Set up some node states
        observer.node_states = {
            "node1": {"status": "running"},
            "node2": {"status": "completed"},
            "node3": {"status": "failed"},
            "node4": {"status": "failed"},
        }

        failed = observer.get_failed_nodes()
        assert set(failed) == {"node3", "node4"}


class TestWebSocketObserver:
    """Test the WebSocket observer."""

    @pytest.mark.asyncio
    async def test_websocket_observer_sends_events(self):
        """Test that WebSocketObserver sends events over WebSocket."""
        # Mock WebSocket handler
        ws_handler = MagicMock()
        observer = WebSocketObserver(ws_handler)

        # Handle an event
        event = NodeStarted(name="test_node", wave_index=1)
        await observer.handle(event)

        # Should have sent JSON
        ws_handler.send_json.assert_called_once()

        # Check the data sent
        sent_data = ws_handler.send_json.call_args[0][0]
        assert sent_data["type"] == "NodeStarted"
        assert sent_data["name"] == "test_node"
        assert sent_data["wave_index"] == 1

    @pytest.mark.asyncio
    async def test_websocket_observer_adds_progress_info(self):
        """Test that WebSocketObserver adds progress information."""
        ws_handler = MagicMock()
        observer = WebSocketObserver(ws_handler)

        # Handle a node completed event
        event = NodeCompleted(
            name="test_node", wave_index=1, result={"output": "test"}, duration_ms=500
        )
        await observer.handle(event)

        # Check progress info was added
        sent_data = ws_handler.send_json.call_args[0][0]
        assert "progress" in sent_data
        assert sent_data["progress"]["type"] == "node_completed"
        assert sent_data["progress"]["node"] == "test_node"

    @pytest.mark.asyncio
    async def test_websocket_observer_handles_errors(self):
        """Test that WebSocketObserver handles WebSocket errors gracefully."""
        # Mock WebSocket handler that raises
        ws_handler = MagicMock()
        ws_handler.send_json.side_effect = RuntimeError("WebSocket closed")

        observer = WebSocketObserver(ws_handler)

        # Should not raise
        event = NodeStarted(name="test", wave_index=1)
        await observer.handle(event)  # Should log error but not raise


class TestObserverIntegration:
    """Test observers working together with ObserverManager."""

    @pytest.mark.asyncio
    async def test_multiple_observers_with_manager(self):
        """Test multiple observers receiving events through manager."""
        manager = ObserverManager()

        # Add different observers
        logging_obs = LoggingObserver()
        metrics_obs = MetricsObserver()
        state_obs = NodeStateObserver()

        manager.attach(logging_obs)
        manager.attach(metrics_obs)
        manager.attach(state_obs)

        # Run a mini pipeline
        await manager.notify(PipelineStarted(name="test", total_nodes=1, total_waves=1))
        await manager.notify(NodeStarted(name="node1", wave_index=1))
        await manager.notify(NodeCompleted(name="node1", wave_index=1, result={}, duration_ms=100))
        await manager.notify(PipelineCompleted(name="test", duration_ms=100))

        # Check that observers got the data
        assert metrics_obs.completed_nodes == 1
        assert state_obs.get_node_state("node1")["status"] == "completed"
