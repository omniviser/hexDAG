"""Comprehensive integration tests for LocalObserverManager.

These tests verify real-world usage scenarios, thread safety, performance,
and integration with various observer patterns.
"""

import asyncio
import gc
import time
from dataclasses import dataclass
from typing import Any

import pytest

from hexai.adapters.local.local_observer_manager import LocalObserverManager
from hexai.core.application.events.events import Event
from hexai.core.application.events.models import ErrorHandler


@dataclass
class TestEvent(Event):
    """Test event for integration tests."""

    message: str
    value: int = 0


@dataclass
class HighPriorityEvent(Event):
    """High priority test event."""

    data: str


@dataclass
class LowPriorityEvent(Event):
    """Low priority test event."""

    data: str


class TestObserver:
    """Test observer that tracks calls."""

    def __init__(self, name: str, delay: float = 0):
        self.name = name
        self.delay = delay
        self.events: list[Event] = []
        self.call_count = 0

    async def handle(self, event: Event) -> None:
        """Handle event with optional delay."""
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        self.events.append(event)
        self.call_count += 1


class SlowObserver:
    """Observer that simulates slow processing."""

    def __init__(self, delay: float = 1.0):
        self.delay = delay
        self.handled = False

    async def handle(self, event: Event) -> None:
        """Slow event handling."""
        await asyncio.sleep(self.delay)
        self.handled = True


class FailingObserver:
    """Observer that always raises an exception."""

    def __init__(self, error_msg: str = "Intentional failure"):
        self.error_msg = error_msg
        self.attempts = 0

    async def handle(self, event: Event) -> None:
        """Always fail."""
        self.attempts += 1
        raise RuntimeError(self.error_msg)


class RecordingErrorHandler(ErrorHandler):
    """Error handler that records errors for testing."""

    def __init__(self):
        self.errors: list[tuple[Exception, dict[str, Any]]] = []

    def handle_error(self, error: Exception, context: dict[str, Any]) -> None:
        """Record errors."""
        self.errors.append((error, context))


@pytest.mark.asyncio
class TestLocalObserverManagerIntegration:
    """Integration tests for LocalObserverManager."""

    async def test_concurrent_observer_execution(self):
        """Test that observers run concurrently up to the limit."""
        manager = LocalObserverManager(max_concurrent_observers=3)

        # Create 5 observers with different delays
        observers = [TestObserver(f"obs_{i}", delay=0.1) for i in range(5)]

        for obs in observers:
            manager.register(obs)

        # Track execution time
        start = time.time()
        await manager.notify(TestEvent(message="concurrent"))
        duration = time.time() - start

        # With max_concurrent=3 and 5 observers at 0.1s each:
        # Should take ~0.2s (2 batches) not 0.5s (sequential)
        assert duration < 0.3
        assert all(obs.call_count == 1 for obs in observers)

    async def test_mixed_sync_async_observers(self):
        """Test mixing synchronous and asynchronous observers."""
        manager = LocalObserverManager(max_sync_workers=2)

        # Track calls
        async_calls = []
        sync_calls = []

        # Async observer function
        async def async_observer(event: Event):
            async_calls.append(event)

        # Sync observer function
        def sync_observer(event: Event):
            sync_calls.append(event)

        # Register both types
        manager.register(async_observer)
        manager.register(sync_observer)
        manager.register(TestObserver("obj_obs"))

        # Send event
        event = TestEvent(message="mixed")
        await manager.notify(event)

        # All should receive the event
        assert len(async_calls) == 1
        assert len(sync_calls) == 1
        assert async_calls[0] == event
        assert sync_calls[0] == event

    async def test_weak_reference_cleanup(self):
        """Test that weak references are cleaned up properly."""
        manager = LocalObserverManager(use_weak_refs=True)

        # Create observer and register
        observer = TestObserver("weak_test")
        obs_id = manager.register(observer)

        # Verify it's registered
        assert len(manager) == 1

        # Send event - should work
        await manager.notify(TestEvent(message="before"))
        assert observer.call_count == 1

        # Delete observer reference
        del observer
        gc.collect()  # Force garbage collection

        # Manager should be empty now
        assert len(manager) == 0

        # Unregister should return False
        assert not manager.unregister(obs_id)

    async def test_strong_reference_for_functions(self):
        """Test that function observers are kept with strong references."""
        manager = LocalObserverManager(use_weak_refs=True)

        calls = []

        # Register lambda (which would be GC'd without strong ref)
        manager.register(lambda e: calls.append(e))

        # Force GC
        gc.collect()

        # Should still work
        await manager.notify(TestEvent(message="lambda"))
        assert len(calls) == 1

    async def test_event_type_filtering(self):
        """Test that observers only receive events they're interested in."""
        manager = LocalObserverManager()

        test_observer = TestObserver("test")
        high_observer = TestObserver("high")
        all_observer = TestObserver("all")

        # Register with different filters
        manager.register(test_observer, event_types=[TestEvent])
        manager.register(high_observer, event_types=[HighPriorityEvent])
        manager.register(all_observer)  # No filter - gets all

        # Send different event types
        await manager.notify(TestEvent(message="test"))
        await manager.notify(HighPriorityEvent(data="high"))
        await manager.notify(LowPriorityEvent(data="low"))

        # Check filtering worked
        assert test_observer.call_count == 1
        assert high_observer.call_count == 1
        assert all_observer.call_count == 3

    async def test_timeout_handling(self):
        """Test that slow observers are properly timed out."""
        error_handler = RecordingErrorHandler()
        manager = LocalObserverManager(observer_timeout=0.1, error_handler=error_handler)

        # Register slow and fast observers
        slow = SlowObserver(delay=0.5)
        fast = TestObserver("fast")

        manager.register(slow)
        manager.register(fast)

        # Notify - should timeout slow observer
        await manager.notify(TestEvent(message="timeout"))

        # Fast observer should complete
        assert fast.call_count == 1

        # Slow observer should timeout
        assert not slow.handled

        # Error should be recorded
        assert len(error_handler.errors) > 0
        assert any(isinstance(e[0], TimeoutError) for e in error_handler.errors)

    async def test_error_isolation(self):
        """Test that failing observers don't affect others."""
        error_handler = RecordingErrorHandler()
        manager = LocalObserverManager(error_handler=error_handler)

        # Register mix of failing and working observers
        failing1 = FailingObserver("error1")
        working = TestObserver("working")
        failing2 = FailingObserver("error2")

        manager.register(failing1)
        manager.register(working)
        manager.register(failing2)

        # Send event
        await manager.notify(TestEvent(message="errors"))

        # Working observer should still receive event
        assert working.call_count == 1

        # Failures should be recorded
        assert len(error_handler.errors) == 2
        assert failing1.attempts == 1
        assert failing2.attempts == 1

    async def test_context_manager_cleanup(self):
        """Test context manager properly cleans up resources."""
        # Sync context manager
        with LocalObserverManager() as manager:
            obs = TestObserver("ctx")
            manager.register(obs)
            assert len(manager) == 1

        # Executor should be shut down
        assert manager._executor_shutdown

        # Async context manager
        async with LocalObserverManager() as manager:
            obs = TestObserver("async_ctx")
            manager.register(obs)
            await manager.notify(TestEvent(message="async"))
            assert obs.call_count == 1

        # Should be cleared
        assert len(manager) == 0

    async def test_high_volume_notifications(self):
        """Test performance with many observers and notifications."""
        manager = LocalObserverManager(max_concurrent_observers=20)

        # Register many observers
        observers = [TestObserver(f"obs_{i}") for i in range(50)]
        for obs in observers:
            manager.register(obs)

        # Send many events
        events = [TestEvent(message=f"event_{i}", value=i) for i in range(100)]

        start = time.time()
        for event in events:
            await manager.notify(event)
        duration = time.time() - start

        # All observers should get all events
        for obs in observers:
            assert obs.call_count == 100
            assert len(obs.events) == 100

        # Should complete reasonably quickly
        assert duration < 5.0

    async def test_observer_priority_with_semaphore(self):
        """Test that concurrency limits work with many observers."""
        manager = LocalObserverManager(max_concurrent_observers=2)

        # Track concurrent executions
        concurrent_count = 0
        max_concurrent = 0
        lock = asyncio.Lock()

        async def tracking_observer(event: Event):
            nonlocal concurrent_count, max_concurrent

            async with lock:
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)

            await asyncio.sleep(0.05)  # Simulate work

            async with lock:
                concurrent_count -= 1

        # Register many observers
        for _ in range(10):
            manager.register(tracking_observer)

        # Send event
        await manager.notify(TestEvent(message="concurrency"))

        # Should never exceed limit
        assert max_concurrent <= 2

    async def test_clear_by_observer_id(self):
        """Test selective observer removal."""
        manager = LocalObserverManager()

        obs1 = TestObserver("obs1")
        obs2 = TestObserver("obs2")
        obs3 = TestObserver("obs3")

        manager.register(obs1)
        id2 = manager.register(obs2)
        manager.register(obs3)

        # Remove specific observer
        assert manager.unregister(id2)

        # Send event
        await manager.notify(TestEvent(message="selective"))

        # Only obs1 and obs3 should receive
        assert obs1.call_count == 1
        assert obs2.call_count == 0
        assert obs3.call_count == 1

        # Clear all
        manager.clear()
        assert len(manager) == 0

    async def test_thread_pool_executor_limits(self):
        """Test that sync observer thread pool respects limits."""
        manager = LocalObserverManager(max_sync_workers=2)

        execution_times = []

        def slow_sync_observer(event: Event):
            start = time.time()
            time.sleep(0.1)  # Block thread
            execution_times.append(time.time() - start)

        # Register more sync observers than thread pool size
        for _ in range(5):
            manager.register(slow_sync_observer)

        # Send event
        start = time.time()
        await manager.notify(TestEvent(message="thread_pool"))
        total_time = time.time() - start

        # With 2 workers and 5 observers at 0.1s each:
        # Should take ~0.3s (3 batches), not 0.1s (all parallel)
        assert 0.25 < total_time < 0.4

    async def test_reentrant_notifications(self):
        """Test that observers can trigger new notifications."""
        manager = LocalObserverManager()

        secondary_calls = []

        async def reentrant_observer(event: Event):
            if isinstance(event, TestEvent) and event.value < 3:
                # Trigger another notification
                await manager.notify(
                    TestEvent(message=f"reentrant_{event.value}", value=event.value + 1)
                )

        async def tracking_observer(event: Event):
            secondary_calls.append(event)

        manager.register(reentrant_observer)
        manager.register(tracking_observer)

        # Start chain
        await manager.notify(TestEvent(message="initial", value=0))

        # Should have cascading notifications
        assert len(secondary_calls) == 4  # 0, 1, 2, 3

    async def test_observer_exception_types(self):
        """Test handling of different exception types."""
        error_handler = RecordingErrorHandler()
        manager = LocalObserverManager(error_handler=error_handler)

        class CustomError(Exception):
            pass

        async def type_error_observer(event: Event):
            raise TypeError("Type mismatch")

        async def value_error_observer(event: Event):
            raise ValueError("Invalid value")

        async def custom_error_observer(event: Event):
            raise CustomError("Custom error")

        manager.register(type_error_observer)
        manager.register(value_error_observer)
        manager.register(custom_error_observer)

        await manager.notify(TestEvent(message="exceptions"))

        # All errors should be captured
        assert len(error_handler.errors) == 3

        error_types = {type(e[0]) for e in error_handler.errors}
        assert TypeError in error_types
        assert ValueError in error_types
        assert CustomError in error_types

    async def test_memory_efficiency(self):
        """Test memory efficiency with weak references."""
        manager = LocalObserverManager(use_weak_refs=True)

        # Create many observers
        observers = []
        for i in range(100):
            obs = TestObserver(f"mem_{i}")
            observers.append(obs)
            manager.register(obs, keep_alive=False)  # Use weak refs

        # Verify all registered
        assert len(manager) == 100

        # Clear local references but keep one
        kept = observers[0]
        observers.clear()
        gc.collect()

        # Most should be cleaned up (kept one plus any that haven't been GC'd yet)
        # The exact count depends on GC timing, so check it's much less than 100
        assert len(manager) < 10

        # Send event - kept observer should receive
        await manager.notify(TestEvent(message="memory"))
        assert kept.call_count == 1

    async def test_custom_observer_ids(self):
        """Test using custom observer IDs."""
        manager = LocalObserverManager()

        obs1 = TestObserver("obs1")
        obs2 = TestObserver("obs2")

        # Register with custom IDs
        id1 = manager.register(obs1, observer_id="custom_1")
        id2 = manager.register(obs2, observer_id="custom_2")

        assert id1 == "custom_1"
        assert id2 == "custom_2"

        # Unregister by custom ID
        assert manager.unregister("custom_1")

        await manager.notify(TestEvent(message="custom_ids"))

        # Only obs2 should receive
        assert obs1.call_count == 0
        assert obs2.call_count == 1
