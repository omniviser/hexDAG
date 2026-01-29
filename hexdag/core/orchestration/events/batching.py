"""Batching utilities for the observer manager.

This module introduces an ``EventBatcher`` helper that buffers events, builds
envelopes, and decides when to flush batched payloads to downstream observers.
It keeps the batching concerns isolated from ``ObserverManager`` so the manager
can focus on registration and invocation logic.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import cast

from .events import Event, NodeFailed, PipelineCompleted

LOGGER = logging.getLogger(__name__)


class BatchFlushReason(StrEnum):
    """Reason codes for why a batch was flushed."""

    SIZE = "size"
    TIME = "time"
    PRIORITY = "priority"
    OVERLOAD = "overload"
    MANUAL = "manual"
    SHUTDOWN = "shutdown"


class OverloadPolicy(StrEnum):
    """Strategy to apply when the in-memory buffer exceeds its capacity."""

    DROP_OLDEST = "drop-oldest"
    DEGRADE_TO_UNBATCHED = "degrade-to-unbatched"


@dataclass(slots=True)
class BatchingConfig:
    """Configuration knobs for event batching semantics."""

    max_batch_size: int = 256
    max_batch_window_ms: float = 50.0
    max_buffer_events: int = 4096
    overload_policy: OverloadPolicy = OverloadPolicy.DROP_OLDEST
    priority_event_types: tuple[type[Event], ...] = (NodeFailed, PipelineCompleted)


@dataclass(slots=True)
class BatchingMetrics:
    """Simple in-memory metrics tracked by the batcher."""

    event_batches_total: int = 0
    event_batch_flush_reason: Counter[str] = field(default_factory=Counter)
    events_dropped_total: int = 0
    events_unbatched_total: int = 0


@dataclass(slots=True)
class EventBatchEnvelope:
    """Metadata-rich container for a batch of events.

    NOTE: Once the event taxonomy work lands, convert this envelope to the
    canonical shape defined there. The marker keeps the linkage visible for a
    future refactor.
    """

    # NOTE(event-taxonomy): align fields with the canonical envelope schema.
    batch_id: str
    sequence_no: int
    created_at: datetime
    events: Sequence[Event]
    flush_reason: BatchFlushReason


FlushCallback = Callable[[EventBatchEnvelope], Awaitable[None]]


class EventBatcher[EventT: Event]:
    """Collects events and flushes them according to configured rules."""

    def __init__(
        self,
        flush: FlushCallback,
        config: BatchingConfig | None = None,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self._flush_cb = flush
        self._config = config or BatchingConfig()
        self._logger = logger or LOGGER

        self._metrics = BatchingMetrics()
        self._buffer: list[EventT] = []
        self._buffer_opened_at: datetime | None = None
        self._sequence = 0
        self._timer_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._closed = False

    @property
    def metrics(self) -> BatchingMetrics:
        """Expose current batching metrics."""

        return self._metrics

    @property
    def has_pending_events(self) -> bool:
        """Whether events are waiting in the buffer."""

        return bool(self._buffer)

    async def add(self, event: EventT) -> None:
        """Enqueue an event and flush if any rule applies."""

        envelopes: list[EventBatchEnvelope] = []

        async with self._lock:
            if self._closed:
                self._logger.debug("Batcher closed; flushing event immediately")
                envelopes.append(self._build_single_event_envelope(event, BatchFlushReason.MANUAL))
            else:
                is_priority = self._is_priority(event)
                self._buffer.append(event)
                self._ensure_buffer_start_timestamp()

                if len(self._buffer) == 1:
                    self._schedule_timer()

                if self._is_over_capacity():
                    envelopes.extend(await self._handle_overload(event))
                elif len(self._buffer) >= self._config.max_batch_size:
                    envelope = await self._prepare_flush_locked(BatchFlushReason.SIZE)
                    if envelope:
                        envelopes.append(envelope)
                elif is_priority:
                    envelope = await self._prepare_flush_locked(BatchFlushReason.PRIORITY)
                    if envelope:
                        envelopes.append(envelope)

        await self._deliver_envelopes(envelopes)

    async def flush(self, reason: BatchFlushReason = BatchFlushReason.MANUAL) -> None:
        """Flush pending events regardless of thresholds."""

        async with self._lock:
            envelope = await self._prepare_flush_locked(reason)

        if envelope:
            await self._flush_cb(envelope)

    async def close(self) -> None:
        """Flush remaining events and mark the batcher as closed."""

        async with self._lock:
            self._closed = True
            envelope = await self._prepare_flush_locked(BatchFlushReason.SHUTDOWN)
            await self._cancel_timer()

        if envelope:
            await self._flush_cb(envelope)

    # Internal helpers -------------------------------------------------

    def _is_priority(self, event: EventT) -> bool:
        return any(isinstance(event, priority) for priority in self._config.priority_event_types)

    def _is_over_capacity(self) -> bool:
        return len(self._buffer) > self._config.max_buffer_events

    async def _handle_overload(self, triggering_event: EventT) -> list[EventBatchEnvelope]:
        """Apply overload policy when the buffer exceeds capacity."""

        policy = self._config.overload_policy
        envelopes: list[EventBatchEnvelope] = []

        if policy is OverloadPolicy.DROP_OLDEST:
            dropped, resolved = self._drop_oldest_until_capacity()
            if dropped:
                self._metrics.events_dropped_total += dropped
                self._logger.warning(
                    "Dropped %s event(s) due to overload (policy=%s)", dropped, policy.value
                )
            if not resolved:
                envelope = await self._prepare_flush_locked(BatchFlushReason.OVERLOAD)
                if envelope:
                    envelopes.append(envelope)
            return envelopes

        if policy is OverloadPolicy.DEGRADE_TO_UNBATCHED:
            latest = self._buffer.pop() if self._buffer else triggering_event
            envelope = await self._prepare_flush_locked(BatchFlushReason.OVERLOAD)
            if envelope:
                envelopes.append(envelope)
            envelopes.append(self._build_single_event_envelope(latest, BatchFlushReason.OVERLOAD))
            self._metrics.events_unbatched_total += 1
            return envelopes

        raise RuntimeError(f"Unsupported overload policy: {policy!r}")

    def _drop_oldest_until_capacity(self) -> tuple[int, bool]:
        to_drop = max(len(self._buffer) - self._config.max_buffer_events, 0)
        if to_drop <= 0:
            return 0, True

        drop_indices: list[int] = []
        for idx, event in enumerate(self._buffer):
            if self._is_priority(event):
                continue
            drop_indices.append(idx)
            if len(drop_indices) == to_drop:
                break

        if len(drop_indices) < to_drop:
            return 0, False

        for idx in reversed(drop_indices):
            self._buffer.pop(idx)

        return to_drop, True

    def _ensure_buffer_start_timestamp(self) -> None:
        if self._buffer_opened_at is None:
            self._buffer_opened_at = datetime.now(tz=UTC)

    def _schedule_timer(self) -> None:
        if self._timer_task is not None:
            return

        window_seconds = max(self._config.max_batch_window_ms / 1000.0, 0)
        if window_seconds == 0:
            return

        self._timer_task = asyncio.create_task(self._timer_flush(window_seconds))

    async def _timer_flush(self, delay: float) -> None:
        envelope: EventBatchEnvelope | None = None

        try:
            await asyncio.sleep(delay)
            async with self._lock:
                if self._buffer:
                    envelope = await self._prepare_flush_locked(BatchFlushReason.TIME)
                else:
                    envelope = None
        finally:
            self._timer_task = None

        if envelope:
            await self._flush_cb(envelope)

    async def _prepare_flush_locked(self, reason: BatchFlushReason) -> EventBatchEnvelope | None:
        if not self._buffer:
            return None

        await self._cancel_timer()

        envelope = EventBatchEnvelope(
            batch_id=uuid.uuid4().hex,
            sequence_no=self._next_sequence(),
            created_at=self._buffer_opened_at or datetime.now(tz=UTC),
            events=cast("Sequence[Event]", tuple(self._buffer)),
            flush_reason=reason,
        )

        self._buffer.clear()
        self._buffer_opened_at = None

        self._record_envelope(envelope)
        return envelope

    async def _cancel_timer(self) -> None:
        if self._timer_task is None:
            return
        task = self._timer_task
        self._timer_task = None
        if task is asyncio.current_task():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    def _build_single_event_envelope(
        self, event: EventT, reason: BatchFlushReason
    ) -> EventBatchEnvelope:
        envelope = EventBatchEnvelope(
            batch_id=uuid.uuid4().hex,
            sequence_no=self._next_sequence(),
            created_at=datetime.now(tz=UTC),
            events=cast("Sequence[Event]", (event,)),
            flush_reason=reason,
        )
        self._record_envelope(envelope)
        return envelope

    async def _deliver_envelopes(self, envelopes: list[EventBatchEnvelope]) -> None:
        for envelope in envelopes:
            await self._flush_cb(envelope)

    def _record_envelope(self, envelope: EventBatchEnvelope) -> None:
        self._metrics.event_batches_total += 1
        self._metrics.event_batch_flush_reason[envelope.flush_reason.value] += 1
        self._logger.debug(
            "Flushing batch %s (seq=%s, size=%s, reason=%s)",
            envelope.batch_id,
            envelope.sequence_no,
            len(envelope.events),
            envelope.flush_reason.value,
        )

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence


__all__ = [
    "BatchFlushReason",
    "OverloadPolicy",
    "BatchingConfig",
    "BatchingMetrics",
    "EventBatchEnvelope",
    "EventBatcher",
]
