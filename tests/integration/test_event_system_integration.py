"""Integration tests for the event system (observers)."""

import pytest

from hexdag.drivers.observer_manager import LocalObserverManager
from hexdag.kernel.orchestration.events import (
    Event,
    NodeCompleted,
    NodeStarted,
)


class TestEventSystemIntegration:
    """Test event system integration with observers."""

    @pytest.fixture
    def observer_manager(self):
        """Create an observer manager with batching disabled for predictable testing."""
        return LocalObserverManager(batching_enabled=False)

    @pytest.mark.asyncio
    async def test_observer_notification(self, observer_manager):
        """Test that observers receive and handle events."""
        events_received = []

        async def test_observer(event: Event) -> None:
            events_received.append(event)

        observer_manager.register(test_observer)

        # Emit events
        event1 = NodeStarted(name="node1", wave_index=1, dependencies=[])
        event2 = NodeCompleted(name="node1", wave_index=1, result={"data": 123}, duration_ms=150.5)

        await observer_manager.notify(event1)
        await observer_manager.notify(event2)

        assert len(events_received) == 2
        assert isinstance(events_received[0], NodeStarted)
        assert isinstance(events_received[1], NodeCompleted)
        assert events_received[0].name == "node1"
        assert events_received[1].result == {"data": 123}

    @pytest.mark.asyncio
    async def test_observer_class_based(self, observer_manager):
        """Test class-based observer with state."""

        class MetricsObserver:
            def __init__(self):
                self.node_count = 0
                self.total_duration = 0.0

            async def handle(self, event: Event) -> None:
                if isinstance(event, NodeStarted):
                    self.node_count += 1
                elif isinstance(event, NodeCompleted):
                    self.total_duration += event.duration_ms

        metrics = MetricsObserver()
        observer_manager.register(metrics.handle)

        # Emit events
        await observer_manager.notify(NodeStarted(name="node1", wave_index=1, dependencies=[]))
        await observer_manager.notify(
            NodeCompleted(name="node1", wave_index=1, result={}, duration_ms=100.0)
        )
        await observer_manager.notify(NodeStarted(name="node2", wave_index=1, dependencies=[]))
        await observer_manager.notify(
            NodeCompleted(name="node2", wave_index=1, result={}, duration_ms=200.0)
        )

        assert metrics.node_count == 2
        assert metrics.total_duration == 300.0

    @pytest.mark.asyncio
    async def test_multiple_observers(self, observer_manager):
        """Test multiple observers receiving events."""
        events_1 = []
        events_2 = []

        async def observer_1(event: Event) -> None:
            events_1.append(event)

        async def observer_2(event: Event) -> None:
            events_2.append(event)

        observer_manager.register(observer_1)
        observer_manager.register(observer_2)

        event = NodeStarted(name="test", wave_index=1, dependencies=[])
        await observer_manager.notify(event)

        assert len(events_1) == 1
        assert len(events_2) == 1
        assert events_1[0] is events_2[0]  # Same event instance

    @pytest.mark.asyncio
    async def test_observer_unsubscribe(self, observer_manager):
        """Test unsubscribing an observer."""
        events = []

        async def test_observer(event: Event) -> None:
            events.append(event)

        observer_id = observer_manager.register(test_observer)

        # First notification
        await observer_manager.notify(NodeStarted(name="test1", wave_index=1, dependencies=[]))
        assert len(events) == 1

        # Unsubscribe (LocalObserverManager uses unregister() method)
        observer_manager.unregister(observer_id)

        # Second notification should not reach the observer
        await observer_manager.notify(NodeStarted(name="test2", wave_index=1, dependencies=[]))
        assert len(events) == 1  # Still 1, not 2
