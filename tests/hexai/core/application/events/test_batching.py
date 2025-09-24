"""Tests for the event batching helper."""

import asyncio

import pytest

from hexai.core.application.events import NodeFailed, NodeStarted
from hexai.core.application.events.batching import (
    BatchFlushReason,
    BatchingConfig,
    EventBatcher,
    OverloadPolicy,
)


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
