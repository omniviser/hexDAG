"""Tests for the event batching helper."""

import asyncio

import pytest

from hexdag.builtin.adapters.local import LocalObserverManager
from hexdag.core.orchestration.events.batching import (
    BatchFlushReason,
    BatchingConfig,
    EventBatcher,
    OverloadPolicy,
)
from hexdag.core.orchestration.events.events import NodeFailed, NodeStarted


class BatchRecordingObserver:
    """Observer capturing batch envelopes for assertions."""

    def __init__(self) -> None:
        self.envelopes: list = []

    async def handle(self, _event) -> None:  # pragma: no cover - interface compliance
        return None

    async def handle_batch(self, envelope) -> None:
        self.envelopes.append(envelope)


class EventRecordingObserver:
    """Observer capturing individual event deliveries."""

    def __init__(self) -> None:
        self.events: list = []

    async def handle(self, event) -> None:
        self.events.append(event)


@pytest.mark.asyncio
async def test_flush_triggers_on_size_threshold():
    envelopes = []

    async def collector(envelope):
        envelopes.append(envelope)

    batcher = EventBatcher(
        collector,
        BatchingConfig(max_batch_size=2, max_batch_window_ms=1000, max_buffer_events=10),
    )

    await batcher.add(NodeStarted(name="n1", wave_index=1))
    await batcher.add(NodeStarted(name="n2", wave_index=1))

    assert len(envelopes) == 1
    envelope = envelopes[0]
    assert len(envelope.events) == 2
    assert envelope.flush_reason is BatchFlushReason.SIZE
    assert batcher.metrics.event_batches_total == 1


@pytest.mark.asyncio
async def test_flush_triggers_on_time_window():
    envelopes = []

    async def collector(envelope):
        envelopes.append(envelope)

    batcher = EventBatcher(
        collector,
        BatchingConfig(max_batch_size=10, max_batch_window_ms=10, max_buffer_events=10),
    )

    await batcher.add(NodeStarted(name="time_node", wave_index=1))
    await asyncio.sleep(0.05)

    assert len(envelopes) == 1
    assert envelopes[0].flush_reason is BatchFlushReason.TIME


@pytest.mark.asyncio
async def test_priority_event_forces_flush():
    envelopes = []

    async def collector(envelope):
        envelopes.append(envelope)

    batcher = EventBatcher(
        collector,
        BatchingConfig(
            max_batch_size=10,
            max_batch_window_ms=1000,
            max_buffer_events=10,
            priority_event_types=(NodeFailed,),
        ),
    )

    await batcher.add(NodeStarted(name="normal", wave_index=1))
    await batcher.add(NodeFailed(name="failure", wave_index=1, error=RuntimeError("boom")))

    assert len(envelopes) == 1
    assert envelopes[0].flush_reason is BatchFlushReason.PRIORITY
    assert len(envelopes[0].events) == 2


@pytest.mark.asyncio
async def test_overload_drop_oldest_discards_events():
    queue: asyncio.Queue = asyncio.Queue()

    async def collector(envelope):
        await queue.put(envelope)

    batcher = EventBatcher(
        collector,
        BatchingConfig(
            max_batch_size=10,
            max_batch_window_ms=1000,
            max_buffer_events=2,
            overload_policy=OverloadPolicy.DROP_OLDEST,
        ),
    )

    await batcher.add(NodeStarted(name="one", wave_index=1))
    await batcher.add(NodeStarted(name="two", wave_index=1))
    await batcher.add(NodeStarted(name="three", wave_index=1))

    # Force manual flush to observe buffer content
    await batcher.flush()
    envelope = await queue.get()

    assert [event.name for event in envelope.events] == ["two", "three"]
    assert batcher.metrics.events_dropped_total == 1


@pytest.mark.asyncio
async def test_overload_degrade_to_unbatched_delivers_latest_immediately():
    envelopes = []

    async def collector(envelope):
        envelopes.append(envelope)

    batcher = EventBatcher(
        collector,
        BatchingConfig(
            max_batch_size=10,
            max_batch_window_ms=1000,
            max_buffer_events=1,
            overload_policy=OverloadPolicy.DEGRADE_TO_UNBATCHED,
        ),
    )

    await batcher.add(NodeStarted(name="first", wave_index=1))
    await batcher.add(NodeStarted(name="second", wave_index=1))

    # Expect two envelopes: first from overload flush, second from unbatched delivery
    assert len(envelopes) == 2
    assert envelopes[0].flush_reason is BatchFlushReason.OVERLOAD
    assert [event.name for event in envelopes[0].events] == ["first"]
    assert envelopes[1].flush_reason is BatchFlushReason.OVERLOAD
    assert [event.name for event in envelopes[1].events] == ["second"]
    assert batcher.metrics.events_unbatched_total == 1


@pytest.mark.asyncio
async def test_close_flushes_remaining_events():
    envelopes = []

    async def collector(envelope):
        envelopes.append(envelope)

    batcher = EventBatcher(
        collector,
        BatchingConfig(max_batch_size=10, max_batch_window_ms=1000, max_buffer_events=10),
    )

    await batcher.add(NodeStarted(name="close", wave_index=1))
    await batcher.close()

    assert len(envelopes) == 1
    assert envelopes[0].flush_reason is BatchFlushReason.SHUTDOWN


@pytest.mark.asyncio
async def test_local_observer_manager_batches_for_batch_capable_observer():
    observer = BatchRecordingObserver()
    manager = LocalObserverManager(
        batching_config=BatchingConfig(
            max_batch_size=2,
            max_batch_window_ms=1000,
            max_buffer_events=10,
        ),
        batching_enabled=True,
    )

    try:
        manager.register(observer)

        await manager.notify(NodeStarted(name="n1", wave_index=1))
        await manager.notify(NodeStarted(name="n2", wave_index=1))

        assert len(observer.envelopes) == 1
        envelope = observer.envelopes[0]
        assert envelope.flush_reason is BatchFlushReason.SIZE
        assert [event.name for event in envelope.events] == ["n1", "n2"]

        metrics = manager.batching_metrics
        assert metrics is not None
        assert metrics.event_batches_total >= 1
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_local_observer_manager_falls_back_to_single_event_delivery_when_no_batch():
    observer = EventRecordingObserver()
    manager = LocalObserverManager(
        batching_config=BatchingConfig(
            max_batch_size=2,
            max_batch_window_ms=1000,
            max_buffer_events=10,
        ),
        batching_enabled=True,
    )

    try:
        manager.register(observer)

        await manager.notify(NodeStarted(name="first", wave_index=1))
        await manager.notify(NodeStarted(name="second", wave_index=1))

        assert [event.name for event in observer.events] == ["first", "second"]
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_local_observer_manager_triggers_priority_flush():
    observer = BatchRecordingObserver()
    manager = LocalObserverManager(
        batching_config=BatchingConfig(
            max_batch_size=10,
            max_batch_window_ms=1000,
            max_buffer_events=10,
        ),
        batching_enabled=True,
    )

    try:
        manager.register(observer)

        await manager.notify(NodeStarted(name="normal", wave_index=1))
        await manager.notify(NodeFailed(name="fail", wave_index=1, error=RuntimeError("boom")))

        assert len(observer.envelopes) == 1
        envelope = observer.envelopes[0]
        assert envelope.flush_reason is BatchFlushReason.PRIORITY
        assert [type(event) for event in envelope.events] == [NodeStarted, NodeFailed]
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_local_observer_manager_degrades_to_unbatched_for_overload():
    batch_observer = BatchRecordingObserver()
    event_observer = EventRecordingObserver()
    manager = LocalObserverManager(
        batching_config=BatchingConfig(
            max_batch_size=10,
            max_batch_window_ms=1000,
            max_buffer_events=1,
            overload_policy=OverloadPolicy.DEGRADE_TO_UNBATCHED,
        ),
        batching_enabled=True,
    )

    try:
        manager.register(batch_observer)
        manager.register(event_observer)

        await manager.notify(NodeStarted(name="first", wave_index=1))
        await manager.notify(NodeStarted(name="second", wave_index=1))

        assert [event.name for event in event_observer.events] == ["first", "second"]
        assert len(batch_observer.envelopes) == 2
        assert batch_observer.envelopes[0].flush_reason is BatchFlushReason.OVERLOAD
        assert [e.name for e in batch_observer.envelopes[0].events] == ["first"]
        assert batch_observer.envelopes[1].flush_reason is BatchFlushReason.OVERLOAD
        assert [e.name for e in batch_observer.envelopes[1].events] == ["second"]
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_local_observer_manager_respects_event_filters_in_batches():
    started_observer = BatchRecordingObserver()
    failed_observer = BatchRecordingObserver()
    manager = LocalObserverManager(
        batching_config=BatchingConfig(
            max_batch_size=10,
            max_batch_window_ms=1000,
            max_buffer_events=10,
        ),
        batching_enabled=True,
    )

    try:
        manager.register(started_observer, event_types=[NodeStarted])
        manager.register(failed_observer, event_types=[NodeFailed])

        await manager.notify(NodeStarted(name="step", wave_index=1))
        await manager.notify(NodeFailed(name="oops", wave_index=1, error=RuntimeError("x")))

        assert len(started_observer.envelopes) == 1
        assert [event.name for event in started_observer.envelopes[0].events] == ["step"]
        assert len(failed_observer.envelopes) == 1
        assert [type(event) for event in failed_observer.envelopes[0].events] == [NodeFailed]
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_local_observer_manager_close_flushes_pending_events():
    observer = BatchRecordingObserver()
    manager = LocalObserverManager(
        batching_config=BatchingConfig(
            max_batch_size=10,
            max_batch_window_ms=1000,
            max_buffer_events=10,
        ),
        batching_enabled=True,
    )

    try:
        manager.register(observer)

        await manager.notify(NodeStarted(name="pending", wave_index=1))
    finally:
        await manager.close()

    assert observer.envelopes
    assert observer.envelopes[-1].flush_reason is BatchFlushReason.SHUTDOWN
