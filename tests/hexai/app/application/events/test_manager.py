"""Tests for pipeline event manager functionality.

This module tests the event system that manages pipeline execution lifecycle events
without Context dependency.
"""

import asyncio
from unittest.mock import MagicMock

from hexai.app.application.events import (
    NodeCompletedEvent,
    NodeFailedEvent,
    NodeStartedEvent,
    PipelineEventManager,
    PipelineStartedEvent,
    ValidationWarningEvent,
)
from hexai.app.application.events.base import SyncObserver
from hexai.app.application.events.observers import LoggingObserver, MetricsObserver
import pytest


class MockObserver(SyncObserver):
    """Mock observer for testing event manager functionality."""

    def __init__(self) -> None:
        self.received_events: list = []
        self.call_count = 0

    def handle_sync(self, event) -> None:
        """Handle events synchronously."""
        self.received_events.append(event)
        self.call_count += 1


class TestPipelineEventManager:
    """Test cases for PipelineEventManager without Context dependency."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.event_manager = PipelineEventManager()
        self.mock_observer = MockObserver()
        self.logging_observer = LoggingObserver()
        self.metrics_observer = MetricsObserver()

    @pytest.mark.asyncio
    async def test_subscribe_and_emit_event(self) -> None:
        """Test subscribing to events and emitting them."""
        # Subscribe mock observer
        self.event_manager.subscribe(self.mock_observer)

        # Create and emit an event
        event = NodeStartedEvent(
            node_name="test_node",
            wave_index=1,
            dependencies=["dep1"],
        )

        await self.event_manager.emit(event)

        # Verify the observer received the event
        assert len(self.mock_observer.received_events) == 1
        assert self.mock_observer.received_events[0] == event
        assert self.mock_observer.call_count == 1

    @pytest.mark.asyncio
    async def test_multiple_observers(self) -> None:
        """Test that multiple observers can receive the same event."""
        mock_observer_2 = MockObserver()

        # Subscribe both observers
        self.event_manager.subscribe(self.mock_observer)
        self.event_manager.subscribe(mock_observer_2)

        # Create and emit an event
        event = PipelineStartedEvent(
            pipeline_name="test_pipeline",
            total_waves=2,
            total_nodes=5,
        )

        await self.event_manager.emit(event)

        # Both observers should have received the event
        assert len(self.mock_observer.received_events) == 1
        assert len(mock_observer_2.received_events) == 1
        assert self.mock_observer.received_events[0] == event
        assert mock_observer_2.received_events[0] == event

    @pytest.mark.asyncio
    async def test_emit_multiple_events(self) -> None:
        """Test emitting multiple events to the same observer."""
        self.event_manager.subscribe(self.mock_observer)

        # Create and emit multiple events
        events = [
            NodeStartedEvent(node_name="node1", wave_index=1),
            NodeStartedEvent(node_name="node2", wave_index=1),
            NodeCompletedEvent(
                node_name="node1",
                wave_index=1,
                result={"test": "result"},
                execution_time=1.5,
            ),
        ]

        for event in events:
            await self.event_manager.emit(event)

        # Verify all events were received
        assert len(self.mock_observer.received_events) == 3
        assert self.mock_observer.call_count == 3
        assert self.mock_observer.received_events == events

    @pytest.mark.asyncio
    async def test_real_observers_integration(self) -> None:
        """Test integration with real observer implementations."""
        # Mock the logger methods to avoid actual logging during tests
        self.logging_observer.logger.log = MagicMock()
        self.logging_observer.logger.error = MagicMock()

        # Subscribe real observers
        self.event_manager.subscribe(self.logging_observer)
        self.event_manager.subscribe(self.metrics_observer)

        # Emit a pipeline started event
        pipeline_event = PipelineStartedEvent(
            pipeline_name="integration_test",
            total_waves=2,
            total_nodes=4,
        )
        await self.event_manager.emit(pipeline_event)

        # Emit a node failed event
        node_event = NodeFailedEvent(
            node_name="failing_node",
            wave_index=1,
            error=RuntimeError("Test error"),
        )
        await self.event_manager.emit(node_event)

        # Verify logging observer received calls
        assert self.logging_observer.logger.log.called
        assert self.logging_observer.logger.error.called

        # Verify metrics observer updated its state
        assert self.metrics_observer.total_nodes == 4
        assert self.metrics_observer.error_count == 1

    @pytest.mark.asyncio
    async def test_observer_error_handling(self) -> None:
        """Test that observer errors don't crash the event manager."""

        class FailingObserver(SyncObserver):
            def handle_sync(self, event) -> None:
                raise RuntimeError("Observer failed")

        failing_observer = FailingObserver()
        self.event_manager.subscribe(failing_observer)
        self.event_manager.subscribe(self.mock_observer)

        # Emit an event - failing observer should not prevent others from working
        event = NodeCompletedEvent(
            node_name="test_node",
            wave_index=1,
            result={"output": "test"},
            execution_time=0.5,
        )

        # This should not raise an exception
        await self.event_manager.emit(event)

        # The working observer should still receive the event
        assert len(self.mock_observer.received_events) == 1
        assert self.mock_observer.received_events[0] == event

    @pytest.mark.asyncio
    async def test_concurrent_event_emission(self) -> None:
        """Test concurrent event emission."""
        self.event_manager.subscribe(self.mock_observer)

        # Create multiple events
        events = [NodeStartedEvent(node_name=f"node_{i}", wave_index=1) for i in range(10)]

        # Emit events concurrently
        await asyncio.gather(*[self.event_manager.emit(event) for event in events])

        # All events should be received
        assert len(self.mock_observer.received_events) == 10

    def test_unsubscribe_observer(self) -> None:
        """Test unsubscribing observers."""
        # Subscribe observer
        self.event_manager.subscribe(self.mock_observer)
        assert len(self.event_manager._global_observers) == 1

        # Unsubscribe observer
        self.event_manager.unsubscribe(self.mock_observer)
        assert len(self.event_manager._global_observers) == 0

    @pytest.mark.asyncio
    async def test_emit_after_unsubscribe(self) -> None:
        """Test that unsubscribed observers don't receive events."""
        # Subscribe and then unsubscribe
        self.event_manager.subscribe(self.mock_observer)
        self.event_manager.unsubscribe(self.mock_observer)

        # Emit an event
        event = ValidationWarningEvent(
            pipeline_name="test_pipeline",
            warnings=["test warning"],
        )
        await self.event_manager.emit(event)

        # Observer should not have received the event
        assert len(self.mock_observer.received_events) == 0
