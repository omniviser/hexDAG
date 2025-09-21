"""Tests for LocalObserverManager adapter."""

import asyncio
from unittest.mock import MagicMock

import pytest

from hexai.adapters.local.local_observer_manager import LocalObserverManager
from hexai.core.application.events.events import Event, NodeCompleted, NodeStarted
from hexai.core.application.events.models import ErrorHandler


class TestObserver:
    """Test observer class."""

    def __init__(self):
        self.events = []
        self.handle_called = False

    async def handle(self, event: Event) -> None:
        """Handle an event."""
        self.events.append(event)
        self.handle_called = True


class TestLocalObserverManager:
    """Test the LocalObserverManager adapter."""

    @pytest.fixture
    def manager(self):
        """Create a LocalObserverManager instance."""
        return LocalObserverManager(
            max_concurrent_observers=5,
            observer_timeout=2.0,
            max_sync_workers=2,
            use_weak_refs=False,
        )

    @pytest.fixture
    def manager_with_weak_refs(self):
        """Create a LocalObserverManager with weak refs enabled."""
        return LocalObserverManager(
            max_concurrent_observers=5,
            observer_timeout=2.0,
            max_sync_workers=2,
            use_weak_refs=True,
        )

    @pytest.mark.asyncio
    async def test_register_and_notify_observer(self, manager):
        """Test registering an observer and receiving notifications."""
        observer = TestObserver()
        observer_id = manager.register(observer)

        assert observer_id is not None
        assert len(manager) == 1

        event = NodeStarted(name="test_node", wave_index=0)
        await manager.notify(event)

        assert observer.handle_called
        assert len(observer.events) == 1
        assert isinstance(observer.events[0], NodeStarted)

    @pytest.mark.asyncio
    async def test_register_function_observer(self, manager):
        """Test registering a function as an observer."""
        events = []

        def sync_observer(event: Event) -> None:
            events.append(event)

        observer_id = manager.register(sync_observer)
        assert observer_id is not None

        event = NodeStarted(name="test_node", wave_index=0)
        await manager.notify(event)

        # Give thread pool time to execute
        await asyncio.sleep(0.1)
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_register_async_function_observer(self, manager):
        """Test registering an async function as an observer."""
        events = []

        async def async_observer(event: Event) -> None:
            events.append(event)

        observer_id = manager.register(async_observer)
        assert observer_id is not None

        event = NodeStarted(name="test_node", wave_index=0)
        await manager.notify(event)

        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_event_type_filtering(self, manager):
        """Test that observers only receive events they're registered for."""
        observer1 = TestObserver()
        observer2 = TestObserver()

        # Register observer1 for NodeStarted only
        manager.register(observer1, event_types=[NodeStarted])

        # Register observer2 for NodeCompleted only
        manager.register(observer2, event_types=[NodeCompleted])

        # Send both event types
        start_event = NodeStarted(name="test_node", wave_index=0)
        complete_event = NodeCompleted(
            name="test_node", wave_index=0, result={"value": 42}, duration_ms=100.0
        )

        await manager.notify(start_event)
        await manager.notify(complete_event)

        # Observer1 should only have received NodeStarted
        assert len(observer1.events) == 1
        assert isinstance(observer1.events[0], NodeStarted)

        # Observer2 should only have received NodeCompleted
        assert len(observer2.events) == 1
        assert isinstance(observer2.events[0], NodeCompleted)

    @pytest.mark.asyncio
    async def test_unregister_observer(self, manager):
        """Test unregistering an observer."""
        observer = TestObserver()
        observer_id = manager.register(observer)

        assert len(manager) == 1

        # Unregister should return True
        assert manager.unregister(observer_id) is True
        assert len(manager) == 0

        # Unregistering again should return False
        assert manager.unregister(observer_id) is False

        # Observer should not receive events after unregistration
        event = NodeStarted(name="test_node", wave_index=0)
        await manager.notify(event)
        assert not observer.handle_called

    @pytest.mark.asyncio
    async def test_clear_observers(self, manager):
        """Test clearing all observers."""
        observer1 = TestObserver()
        observer2 = TestObserver()

        manager.register(observer1)
        manager.register(observer2)

        assert len(manager) == 2

        manager.clear()
        assert len(manager) == 0

        # No observers should receive events
        event = NodeStarted(name="test_node", wave_index=0)
        await manager.notify(event)

        assert not observer1.handle_called
        assert not observer2.handle_called

    @pytest.mark.asyncio
    async def test_error_handling(self, manager):
        """Test that observer errors don't crash the pipeline."""
        error_handler = MagicMock(spec=ErrorHandler)
        manager = LocalObserverManager(error_handler=error_handler, observer_timeout=0.5)

        # Observer that raises an error
        async def failing_observer(event: Event) -> None:
            raise ValueError("Test error")

        # Observer that works
        working_observer = TestObserver()

        manager.register(failing_observer)
        manager.register(working_observer)

        event = NodeStarted(name="test_node", wave_index=0)
        await manager.notify(event)

        # Working observer should still receive the event
        assert working_observer.handle_called

        # Error handler should have been called
        error_handler.handle_error.assert_called()

    @pytest.mark.asyncio
    async def test_timeout_handling(self, manager):
        """Test that slow observers are handled with timeout."""
        error_handler = MagicMock(spec=ErrorHandler)
        manager = LocalObserverManager(
            error_handler=error_handler,
            observer_timeout=0.1,  # Very short timeout
        )

        # Observer that takes too long
        async def slow_observer(event: Event) -> None:
            await asyncio.sleep(1.0)  # Longer than timeout

        # Fast observer
        fast_observer = TestObserver()

        manager.register(slow_observer)
        manager.register(fast_observer)

        event = NodeStarted(name="test_node", wave_index=0)
        await manager.notify(event)

        # Fast observer should still receive the event
        assert fast_observer.handle_called

        # Error handler should have been called for timeout
        error_handler.handle_error.assert_called()

    @pytest.mark.asyncio
    async def test_weak_reference_cleanup(self, manager_with_weak_refs):
        """Test that weak references are cleaned up properly."""

        class WeakObserver:
            async def handle(self, event: Event) -> None:
                pass

        observer = WeakObserver()
        observer_id = manager_with_weak_refs.register(observer)

        assert len(manager_with_weak_refs) == 1

        # Delete the observer (weak ref should be cleaned up on next notify)
        del observer

        # Trigger cleanup through notify
        event = NodeStarted(name="test_node", wave_index=0)
        await manager_with_weak_refs.notify(event)

        # The weak reference should be gone
        assert observer_id not in manager_with_weak_refs._weak_handlers

    @pytest.mark.asyncio
    async def test_keep_alive_parameter(self, manager_with_weak_refs):
        """Test that keep_alive parameter maintains strong references."""

        class WeakObserver:
            async def handle(self, event: Event) -> None:
                pass

        observer = WeakObserver()
        observer_id = manager_with_weak_refs.register(observer, keep_alive=True)

        assert len(manager_with_weak_refs) == 1

        # Delete the local reference
        del observer

        # Trigger notify - observer should still be there due to keep_alive
        event = NodeStarted(name="test_node", wave_index=0)
        await manager_with_weak_refs.notify(event)

        # Observer should still be in strong refs
        assert observer_id in manager_with_weak_refs._strong_refs
        assert len(manager_with_weak_refs) == 1

    @pytest.mark.asyncio
    async def test_context_manager(self, manager):
        """Test using the adapter as a context manager."""
        with manager as mgr:
            observer = TestObserver()
            mgr.register(observer)
            assert len(mgr) == 1

        # After exiting, thread pool should be shut down
        assert manager._executor_shutdown is True

    @pytest.mark.asyncio
    async def test_async_context_manager(self, manager):
        """Test using the adapter as an async context manager."""
        async with manager as mgr:
            observer = TestObserver()
            mgr.register(observer)
            assert len(mgr) == 1

            event = NodeStarted(name="test_node", wave_index=0)
            await mgr.notify(event)

            assert observer.handle_called

        # After exiting, resources should be cleaned up
        assert manager._executor_shutdown is True

    @pytest.mark.asyncio
    async def test_concurrent_observer_limit(self):
        """Test that concurrent observer limit is respected."""
        manager = LocalObserverManager(
            max_concurrent_observers=2,  # Only 2 concurrent
            observer_timeout=5.0,
        )

        execution_order = []

        async def tracked_observer(name: str, delay: float):
            async def observer(event: Event) -> None:
                execution_order.append(f"{name}_start")
                await asyncio.sleep(delay)
                execution_order.append(f"{name}_end")

            return observer

        # Register 4 observers
        manager.register(await tracked_observer("A", 0.1))
        manager.register(await tracked_observer("B", 0.1))
        manager.register(await tracked_observer("C", 0.1))
        manager.register(await tracked_observer("D", 0.1))

        event = NodeStarted(name="test_node", wave_index=0)
        await manager.notify(event)

        # With max_concurrent=2, we should see at most 2 observers starting
        # before any complete
        max_concurrent = 0
        current_running = 0

        for item in execution_order:
            if item.endswith("_start"):
                current_running += 1
                max_concurrent = max(max_concurrent, current_running)
            else:
                current_running -= 1

        # Should not exceed the limit
        assert max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_parameters_passed_correctly(self):
        """Test that all constructor parameters are passed to the inner ObserverManager."""
        custom_error_handler = MagicMock(spec=ErrorHandler)

        manager = LocalObserverManager(
            max_concurrent_observers=15,
            observer_timeout=10.0,
            max_sync_workers=8,
            error_handler=custom_error_handler,
            use_weak_refs=False,
        )

        # Check that parameters were stored correctly
        assert manager._max_concurrent == 15
        assert manager._timeout == 10.0
        assert manager._error_handler == custom_error_handler
        assert manager._use_weak_refs is False
        assert manager._executor._max_workers == 8
