import asyncio
from typing import Any

import pytest

from hexai.core.application.events.context import reset_observer_manager, set_observer_manager
from hexai.core.application.events.decorators import (
    control_handler,
    emits_events,
    observer,
)
from hexai.core.application.events.events import (
    LifecycleEventCompleted,
    LifecycleEventFailed,
    LifecycleEventStarted,
)
from hexai.core.registry.decorators import node


class DummyObserverManager:
    """Simple stand-in for ObserverManagerPort collecting notified events."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    async def notify(self, event: Any) -> None:
        self.events.append(event)


@pytest.mark.asyncio
async def test_emits_events_async_success() -> None:
    manager = DummyObserverManager()
    token = set_observer_manager(manager)
    try:

        @emits_events(include_args=("value",), include_result=True)
        async def sample(value: int) -> int:
            await asyncio.sleep(0)
            return value + 1

        result = await sample(41)
    finally:
        reset_observer_manager(token)

    assert result == 42
    assert len(manager.events) == 2
    start, completed = manager.events
    assert isinstance(start, LifecycleEventStarted)
    assert start.component_type == "function"
    assert start.payload == {"value": 41}
    assert isinstance(completed, LifecycleEventCompleted)
    assert completed.result == 42
    assert completed.duration_ms is not None


@pytest.mark.asyncio
async def test_emits_events_async_failure() -> None:
    manager = DummyObserverManager()
    token = set_observer_manager(manager)
    try:

        @emits_events()
        async def boom() -> None:
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError):
            await boom()
    finally:
        reset_observer_manager(token)

    assert len(manager.events) == 2
    start, failed = manager.events
    assert isinstance(start, LifecycleEventStarted)
    assert isinstance(failed, LifecycleEventFailed)
    assert failed.error == "fail"
    assert failed.exception_type == "RuntimeError"


@pytest.mark.asyncio
async def test_emits_events_uses_decorator_metadata() -> None:
    manager = DummyObserverManager()
    token = set_observer_manager(manager)
    try:

        @control_handler(name="retry_policy", priority=5)
        @emits_events()
        async def retry_policy() -> None:
            await asyncio.sleep(0)

        await retry_policy()
    finally:
        reset_observer_manager(token)

    assert manager.events
    start_event = manager.events[0]
    assert isinstance(start_event, LifecycleEventStarted)
    assert start_event.component_type == "policy"
    assert start_event.component_name == "retry_policy"
    assert start_event.metadata.get("priority") == 5


@pytest.mark.asyncio
async def test_emits_events_on_node_method() -> None:
    manager = DummyObserverManager()
    token = set_observer_manager(manager)
    try:

        @node(name="demo_node", namespace="test")
        class DemoNode:
            @emits_events(include_args=("value",))
            async def execute(self, value: int) -> int:
                return value * 2

        demo = DemoNode()
        await demo.execute(21)
    finally:
        reset_observer_manager(token)

    assert len(manager.events) >= 2
    start_event = manager.events[0]
    assert isinstance(start_event, LifecycleEventStarted)
    assert start_event.component_type == "node"
    assert start_event.component_name == "demo_node"
    assert start_event.metadata.get("namespace") == "test"
    assert start_event.payload == {"value": 21}


def test_emits_events_sync_function() -> None:
    manager = DummyObserverManager()
    token = set_observer_manager(manager)
    try:

        @observer(id="sync-observer")
        @emits_events(include_result=True)
        def sync_handler(value: int) -> int:
            return value + 5

        output = sync_handler(5)
    finally:
        reset_observer_manager(token)

    assert output == 10
    assert len(manager.events) == 2
    start_event, completed_event = manager.events
    assert isinstance(start_event, LifecycleEventStarted)
    assert start_event.component_type == "observer"
    assert isinstance(completed_event, LifecycleEventCompleted)
    assert completed_event.result == 10
