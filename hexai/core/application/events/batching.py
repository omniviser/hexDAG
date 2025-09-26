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
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import cast

from .events import Event


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
    SPILL_TO_DISK = "spill-to-disk"
    DEGRADE_TO_UNBATCHED = "degrade-to-unbatched"


@dataclass(slots=True)
class BatchingConfig:
    """Configuration knobs for event batching semantics."""

    max_batch_size: int = 256
    max_batch_window_ms: float = 50.0
    max_buffer_events: int = 4096
    overload_policy: OverloadPolicy = OverloadPolicy.DROP_OLDEST
    priority_event_types: tuple[type[Event], ...] = ()
    spill_directory: str | None = None


@dataclass(slots=True)
class BatchingMetrics:
    """Simple in-memory metrics tracked by the batcher."""

    event_batches_total: int = 0
    event_batch_flush_reason: Counter[str] = field(default_factory=Counter)
    events_dropped_total: int = 0
    events_spilled_total: int = 0
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
        loop: asyncio.AbstractEventLoop | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._flush_cb = flush
        self._config = config or BatchingConfig()
        self._loop = loop or asyncio.get_event_loop()
        self._logger = logger or logging.getLogger(__name__)

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

        async with self._lock:
            if self._closed:
                self._logger.debug("Batcher closed; flushing event immediately")
                await self._deliver_immediately(event)
                return

            is_priority = self._is_priority(event)
            self._buffer.append(event)
            self._ensure_buffer_start_timestamp()

            if len(self._buffer) == 1:
                self._schedule_timer()

            if self._is_over_capacity():
                await self._handle_overload(event)
                return

            if len(self._buffer) >= self._config.max_batch_size:
                await self._flush_locked(BatchFlushReason.SIZE)
                return

            if is_priority:
                await self._flush_locked(BatchFlushReason.PRIORITY)

    async def flush(self, reason: BatchFlushReason = BatchFlushReason.MANUAL) -> None:
        """Flush pending events regardless of thresholds."""

        async with self._lock:
            await self._flush_locked(reason)

    async def close(self) -> None:
        """Flush remaining events and mark the batcher as closed."""

        async with self._lock:
            self._closed = True
            await self._flush_locked(BatchFlushReason.SHUTDOWN)
            await self._cancel_timer()

    # Internal helpers -------------------------------------------------

    def _is_priority(self, event: EventT) -> bool:
        return any(isinstance(event, priority) for priority in self._config.priority_event_types)

    def _is_over_capacity(self) -> bool:
        return len(self._buffer) > self._config.max_buffer_events

    async def _handle_overload(self, triggering_event: EventT) -> None:
        """Apply overload policy when the buffer exceeds capacity."""

        policy = self._config.overload_policy
        if policy is OverloadPolicy.DROP_OLDEST:
            dropped, resolved = self._drop_oldest_until_capacity()
            self._metrics.events_dropped_total += dropped
            self._logger.warning(
                "Dropped %s event(s) due to overload (policy=%s)", dropped, policy.value
            )
            if not resolved:
                await self._flush_locked(BatchFlushReason.OVERLOAD)
            return

        if policy is OverloadPolicy.SPILL_TO_DISK:
            await self._spill_buffer_to_disk()
            await self._flush_locked(BatchFlushReason.OVERLOAD)
            return

        if policy is OverloadPolicy.DEGRADE_TO_UNBATCHED:
            latest = self._buffer.pop() if self._buffer else triggering_event
            await self._flush_locked(BatchFlushReason.OVERLOAD)
            await self._deliver_immediately(latest)
            self._metrics.events_unbatched_total += 1
            return

    def _drop_oldest_until_capacity(self) -> tuple[int, bool]:
        to_drop = max(len(self._buffer) - self._config.max_buffer_events, 0)
        if to_drop <= 0:
            return 0, True

        dropped = 0
        kept: list[EventT] = []
        for event in self._buffer:
            if dropped < to_drop and not self._is_priority(event):
                dropped += 1
                continue
            kept.append(event)

        fully_resolved = dropped == to_drop
        if not fully_resolved and len(kept) > self._config.max_buffer_events:
            # When all buffered events are priority we can't drop safely.
            kept = list(self._buffer)
            dropped = 0
        self._buffer = kept
        return dropped, fully_resolved

    async def _spill_buffer_to_disk(self) -> None:
        """Write buffered events to disk when configured to do so."""

        if not self._config.spill_directory:
            # Without a directory, fall back to drop-oldest semantics.
            dropped, _ = self._drop_oldest_until_capacity()
            self._metrics.events_dropped_total += dropped
            self._logger.warning(
                "Spill-to-disk requested but no directory configured; dropped %s event(s)",
                dropped,
            )
            return

        path = self._build_spill_path()
        await self._write_spill_file(path, self._buffer)
        self._metrics.events_spilled_total += len(self._buffer)
        self._logger.warning("Spilled %s event(s) to %s", len(self._buffer), path)
        self._buffer.clear()
        self._buffer_opened_at = None

    async def _write_spill_file(self, path: str, events: Iterable[EventT]) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._write_spill_file_sync, path, list(events))

    def _write_spill_file_sync(self, path: str, events: list[EventT]) -> None:
        import json
        from pathlib import Path

        payload = [self._serialize_event(event) for event in events]
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    def _serialize_event(self, event: EventT) -> dict[str, object]:
        """Best-effort JSON serialization for spill files."""

        # NOTE(event-taxonomy): once canonical envelope exists, reuse its serializer.
        data = event.__dict__.copy()
        data["__event_type__"] = type(event).__name__
        return data

    def _build_spill_path(self) -> str:
        from pathlib import Path

        directory = Path(self._config.spill_directory or ".")
        directory.mkdir(parents=True, exist_ok=True)
        return str(directory / f"event-spill-{uuid.uuid4().hex}.json")

    def _ensure_buffer_start_timestamp(self) -> None:
        if self._buffer_opened_at is None:
            self._buffer_opened_at = datetime.now(tz=UTC)

    def _schedule_timer(self) -> None:
        if self._timer_task is not None:
            return

        window_seconds = max(self._config.max_batch_window_ms / 1000.0, 0)
        if window_seconds == 0:
            return

        self._timer_task = self._loop.create_task(self._timer_flush(window_seconds))

    async def _timer_flush(self, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
            async with self._lock:
                if self._buffer:
                    await self._flush_locked(BatchFlushReason.TIME)
        finally:
            self._timer_task = None

    async def _flush_locked(self, reason: BatchFlushReason) -> None:
        if not self._buffer:
            return

        await self._cancel_timer()

        self._sequence += 1
        envelope = EventBatchEnvelope(
            batch_id=uuid.uuid4().hex,
            sequence_no=self._sequence,
            created_at=self._buffer_opened_at or datetime.now(tz=UTC),
            events=cast("Sequence[Event]", tuple(self._buffer)),
            flush_reason=reason,
        )

        self._buffer = []
        self._buffer_opened_at = None

        self._metrics.event_batches_total += 1
        self._metrics.event_batch_flush_reason[reason.value] += 1

        self._logger.debug(
            "Flushing batch %s (seq=%s, size=%s, reason=%s)",
            envelope.batch_id,
            envelope.sequence_no,
            len(envelope.events),
            reason.value,
        )

        await self._flush_cb(envelope)

    async def _cancel_timer(self) -> None:
        if self._timer_task is None:
            return
        task = self._timer_task
        self._timer_task = None
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _deliver_immediately(self, event: EventT) -> None:
        envelope = EventBatchEnvelope(
            batch_id=uuid.uuid4().hex,
            sequence_no=self._sequence + 1,
            created_at=datetime.now(tz=UTC),
            events=cast("Sequence[Event]", (event,)),
            flush_reason=BatchFlushReason.MANUAL,
        )
        self._sequence += 1
        self._metrics.event_batches_total += 1
        self._metrics.event_batch_flush_reason[BatchFlushReason.MANUAL.value] += 1
        self._logger.debug(
            "Delivering single-event batch %s (seq=%s)", envelope.batch_id, envelope.sequence_no
        )
        await self._flush_cb(envelope)


__all__ = [
    "BatchFlushReason",
    "OverloadPolicy",
    "BatchingConfig",
    "BatchingMetrics",
    "EventBatchEnvelope",
    "EventBatcher",
]
