"""Tests for the ObserverManager - the clean event manager implementation."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from hexai.core.application.events.events import NodeStarted, PipelineStarted
from hexai.core.application.events.manager import ObserverManager


class TestObserverManager:
    """Test the ObserverManager for fire-and-forget observability."""

    @pytest.mark.asyncio
    async def test_register_and_notify(self):
        """Test registering observers and notifying them of events."""
        manager = ObserverManager()

        # Create mock observer
        mock_observer = MagicMock()
        mock_observer.handle = AsyncMock()

        # Register observer
        manager.register(mock_observer)
        assert len(manager) == 1

        # Emit event
        event = NodeStarted(name="test_node", wave_index=1)
        await manager.notify(event)

        # Verify observer was called
        mock_observer.handle.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_multiple_observers_receive_events(self):
        """Test that multiple observers all receive the same event."""
        manager = ObserverManager()

        # Create multiple observers
        observers = []
        for _ in range(3):
            obs = MagicMock()
            obs.handle = AsyncMock()
            observers.append(obs)
            manager.register(obs)

        assert len(manager) == 3

        # Emit event
        event = PipelineStarted(name="test_pipeline", total_nodes=5, total_waves=2)
        await manager.notify(event)

        # All observers should be called
        for obs in observers:
            obs.handle.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_observer_error_isolation(self):
        """Test that errors in one observer don't affect others."""
        manager = ObserverManager()

        # Create failing observer
        failing_observer = MagicMock()
        failing_observer.handle = AsyncMock(side_effect=RuntimeError("Observer failed"))

        # Create working observer
        working_observer = MagicMock()
        working_observer.handle = AsyncMock()

        # Register both
        manager.register(failing_observer)
        manager.register(working_observer)

        # Emit event - should not raise
        event = NodeStarted(name="test", wave_index=1)
        await manager.notify(event)

        # Working observer should still be called
        working_observer.handle.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_async_function_as_observer(self):
        """Test that async functions can be used as observers."""
        manager = ObserverManager()

        # Track calls
        calls = []

        async def async_observer(event):
            calls.append(event)

        manager.register(async_observer)

        # Emit event
        event = NodeStarted(name="test", wave_index=1)
        await manager.notify(event)

        # Should be called
        assert len(calls) == 1
        assert calls[0] == event

    @pytest.mark.asyncio
    async def test_sync_function_as_observer(self):
        """Test that sync functions can be used as observers."""
        manager = ObserverManager()

        # Track calls
        calls = []

        def sync_observer(event):
            calls.append(event)

        manager.register(sync_observer)

        # Emit event
        event = NodeStarted(name="test", wave_index=1)
        await manager.notify(event)

        # Give time for executor
        await asyncio.sleep(0.1)

        # Should be called
        assert len(calls) == 1
        assert calls[0] == event

    def test_unregister_observer(self):
        """Test removing observers."""
        manager = ObserverManager()

        obs1 = MagicMock()
        obs2 = MagicMock()

        # Register returns IDs now
        id1 = manager.register(obs1)
        _ = manager.register(obs2)
        assert len(manager) == 2

        # Unregister one by ID
        result = manager.unregister(id1)
        assert result is True
        assert len(manager) == 1

        # Unregister non-existent ID (should return False)
        result = manager.unregister(id1)
        assert result is False
        assert len(manager) == 1

    def test_clear_all_observers(self):
        """Test clearing all observers."""
        manager = ObserverManager()

        # Add multiple observers
        for _ in range(5):
            manager.register(MagicMock())

        assert len(manager) == 5

        # Clear all
        manager.clear()
        assert len(manager) == 0

    @pytest.mark.asyncio
    async def test_empty_manager_does_nothing(self):
        """Test that notifying with no observers doesn't error."""
        manager = ObserverManager()

        # Should not raise
        await manager.notify(NodeStarted(name="test", wave_index=1))

    @pytest.mark.asyncio
    async def test_concurrent_notifications(self):
        """Test that concurrent notifications work correctly."""
        manager = ObserverManager()

        # Track all calls
        calls = []

        async def tracking_observer(event):
            calls.append(event)
            await asyncio.sleep(0.01)  # Simulate some work

        manager.register(tracking_observer)

        # Create multiple events
        events = [NodeStarted(name=f"node_{i}", wave_index=1) for i in range(5)]

        # Notify concurrently
        await asyncio.gather(*[manager.notify(event) for event in events])

        # All events should be tracked
        assert len(calls) == 5
        assert {e.name for e in calls} == {f"node_{i}" for i in range(5)}
