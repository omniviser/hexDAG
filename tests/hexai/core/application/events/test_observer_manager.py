"""Tests for the ObserverManager - the clean observer manager implementation."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from hexai.core.application.events import NodeStarted, ObserverManager, PipelineStarted, observer


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

        # Add multiple observers (keep references to prevent GC)
        observers = []
        for _ in range(5):
            mock = MagicMock()
            observers.append(mock)  # Keep reference
            manager.register(mock)

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

    @pytest.mark.asyncio
    async def test_observer_receives_only_registered_events(self):
        """Test that observers only receive events they registered for."""
        manager = ObserverManager()

        # Create mock observers
        node_observer = AsyncMock()
        pipeline_observer = AsyncMock()
        all_observer = AsyncMock()

        # Register with different filters
        from hexai.core.application.events import (
            NodeCompleted,
            NodeFailed,
            PipelineCompleted,
            PipelineStarted,
            ToolCalled,
        )

        manager.register(node_observer, event_types=[NodeStarted, NodeCompleted, NodeFailed])
        manager.register(pipeline_observer, event_types=[PipelineStarted, PipelineCompleted])
        manager.register(all_observer)  # No filter = all events

        # Send various events
        await manager.notify(NodeStarted(name="test", wave_index=1))
        await manager.notify(PipelineStarted(name="pipeline", total_waves=2, total_nodes=5))
        await manager.notify(ToolCalled(node_name="test", tool_name="api", params={}))

        # Check calls
        assert node_observer.handle.call_count == 1  # Only NodeStarted
        assert pipeline_observer.handle.call_count == 1  # Only PipelineStarted
        assert all_observer.handle.call_count == 3  # All three events

    @pytest.mark.asyncio
    async def test_observer_no_filter_receives_all(self):
        """Test that observer without filter receives all events."""
        manager = ObserverManager()

        observer = AsyncMock()
        manager.register(observer)  # No event_types specified

        # Send various events
        from hexai.core.application.events import PipelineCompleted, ToolCalled

        await manager.notify(NodeStarted(name="test", wave_index=1))
        await manager.notify(ToolCalled(node_name="test", tool_name="api", params={}))
        await manager.notify(PipelineCompleted(name="pipeline", duration_ms=1000))

        assert observer.handle.call_count == 3

    @pytest.mark.asyncio
    async def test_observer_empty_filter_list(self):
        """Test that empty event_types list means no events."""
        manager = ObserverManager()

        observer = AsyncMock()
        manager.register(observer, event_types=[])  # Empty list

        # Send events
        await manager.notify(NodeStarted(name="test", wave_index=1))
        await manager.notify(PipelineStarted(name="pipeline", total_waves=1, total_nodes=1))

        # Should not be called
        assert observer.handle.call_count == 0

    @pytest.mark.asyncio
    async def test_filtering_improves_performance(self):
        """Test that filtering reduces unnecessary invocations."""
        manager = ObserverManager()

        from hexai.core.application.events import ToolCalled

        # Create 10 observers, each interested in only one event type
        node_observers = [AsyncMock() for _ in range(10)]
        tool_observers = [AsyncMock() for _ in range(10)]

        for obs in node_observers:
            manager.register(obs, event_types=[NodeStarted])

        for obs in tool_observers:
            manager.register(obs, event_types=[ToolCalled])

        # Send a NodeStarted event
        await manager.notify(NodeStarted(name="test", wave_index=1))

        # Only node observers should be called
        for obs in node_observers:
            assert obs.handle.call_count == 1

        for obs in tool_observers:
            assert obs.handle.call_count == 0

    @pytest.mark.asyncio
    async def test_weak_ref_cleanup(self):
        """Test that observers are automatically cleaned up when dereferenced."""
        import gc
        import weakref

        manager = ObserverManager(use_weak_refs=True)

        # Create an observer object
        class TestObserver:
            async def handle(self, event):
                pass

        observer = TestObserver()
        weak_ref = weakref.ref(observer)

        # Register the observer
        manager.register(observer)

        # Verify it's registered
        assert len(manager._weak_handlers) == 1

        # Delete the observer
        del observer

        # Force garbage collection
        gc.collect()

        # The weak reference should be dead
        assert weak_ref() is None

        # The manager should auto-clean on next operation
        await manager.notify(NodeStarted(name="test", wave_index=0))

        # Weak dict should be empty now
        assert len(manager._weak_handlers) == 0

    @pytest.mark.asyncio
    async def test_function_kept_alive_with_weak_refs(self):
        """Test that wrapped functions are kept alive."""
        import gc

        manager = ObserverManager(use_weak_refs=True)

        called = []

        # Register a function (will be wrapped)
        def observer_func(event):
            called.append(event)

        obs_id = manager.register(observer_func)

        # Function should be in strong refs (kept alive)
        assert obs_id in manager._strong_refs

        # Delete original function reference
        del observer_func
        gc.collect()

        # Should still work because wrapper keeps it alive
        await manager.notify(NodeStarted(name="test", wave_index=0))

        # Verify it was called
        assert len(called) == 1

    @pytest.mark.asyncio
    async def test_explicit_keep_alive_flag(self):
        """Test that keep_alive flag forces strong reference."""
        import gc
        import weakref

        manager = ObserverManager(use_weak_refs=True)

        class TestObserver:
            async def handle(self, event):
                pass

        observer = TestObserver()
        weak_ref = weakref.ref(observer)

        # Register with keep_alive=True
        obs_id = manager.register(observer, keep_alive=True)

        # Should be in strong refs
        assert obs_id in manager._strong_refs

        # Delete the observer
        del observer
        gc.collect()

        # Should still be alive due to strong ref
        assert weak_ref() is not None

        # Should still work
        await manager.notify(NodeStarted(name="test", wave_index=0))

    def test_no_weak_refs_mode(self):
        """Test that manager works without weak refs."""
        obs_manager = ObserverManager(use_weak_refs=False)

        class TestObserver:
            async def handle(self, event):
                pass

        observer = TestObserver()
        obs_id = obs_manager.register(observer)

        # Should be in normal handlers dict
        assert obs_id in obs_manager._handlers
        assert not hasattr(obs_manager, "_weak_handlers")

    @pytest.mark.asyncio
    async def test_bound_method_fallback(self):
        """Test that bound methods fall back to strong refs."""
        manager = ObserverManager(use_weak_refs=True)

        class TestClass:
            def __init__(self):
                self.called = False

            async def handle(self, event):
                self.called = True

        instance = TestClass()

        # Register bound method (can't be weakly referenced)
        obs_id = manager.register(instance.handle)

        # Should fall back to strong reference
        assert obs_id in manager._strong_refs

        # Should work normally
        await manager.notify(NodeStarted(name="test", wave_index=0))
        assert instance.called

    def test_memory_leak_prevention(self):
        """Test that weak refs actually prevent memory leaks."""
        import gc
        import weakref

        manager = ObserverManager(use_weak_refs=True)

        # Track memory usage
        observers = []
        refs = []

        # Create many observers
        for i in range(100):

            class TempObserver:
                def __init__(self, num):
                    self.num = num
                    self.data = "x" * 1000  # Some data

                async def handle(self, event):
                    pass

            obs = TempObserver(i)
            observers.append(obs)
            refs.append(weakref.ref(obs))
            manager.register(obs)

        # All should be registered
        assert len(manager._weak_handlers) == 100

        # Clear observers list
        observers.clear()
        gc.collect()

        # Most weak refs should be dead (allowing for 1-2 due to Python GC quirks)
        alive_count = sum(1 for ref in refs if ref() is not None)
        assert alive_count <= 2  # Allow for minor GC delays


@pytest.mark.asyncio
async def test_observer_decorator_applies_metadata():
    """Decorated observers should use metadata for registration."""
    manager = ObserverManager()
    calls: list[NodeStarted] = []

    @observer(event_types=[NodeStarted], timeout=0.2, max_concurrency=1, id="logger")
    async def decorated(event):
        calls.append(event)

    observer_id = manager.register(decorated)
    assert observer_id == "logger"
    assert manager._event_filters[observer_id] == {NodeStarted}
    assert manager._observer_timeouts[observer_id] == 0.2
    assert observer_id in manager._observer_semaphores

    await manager.notify(NodeStarted(name="node", wave_index=1))
    await manager.notify(PipelineStarted(name="pipe", total_nodes=1, total_waves=1))

    assert len(calls) == 1
    assert isinstance(manager._observer_semaphores[observer_id], asyncio.Semaphore)

def test_observer_metadata_overrides():
    """Explicit kwargs should override decorator defaults for observers."""
    manager = ObserverManager()

    @observer(event_types=[NodeStarted], timeout=2.0, id="decorated")
    def decorated(event):
        return None

    observer_id = manager.register(
        decorated,
        observer_id="override",
        timeout=0.5,
        event_types=[PipelineStarted],
        max_concurrency=2,
    )

    assert observer_id == "override"
    assert manager._event_filters[observer_id] == {PipelineStarted}
    assert manager._observer_timeouts[observer_id] == 0.5
    semaphore = manager._observer_semaphores[observer_id]
    assert isinstance(semaphore, asyncio.Semaphore)
    assert getattr(semaphore, "_value", None) == 2
