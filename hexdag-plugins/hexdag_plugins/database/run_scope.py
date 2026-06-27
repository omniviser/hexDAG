"""Run-scoped resource management — the dual-mode primitive.

A :class:`RunScopedResource` answers "give me the resource scoped to this
pipeline run, or a standalone one if I'm outside a pipeline" — the pattern
every transactional adapter otherwise reimplements by hand (shared session,
concurrency lock, failure flag, commit-or-rollback decision).

Two modes, selected implicitly by the execution context:

- **Standalone** (no active run): each ``aget()`` creates a fresh resource
  and finalizes it immediately on exit (commit on success, rollback on
  exception).
- **Run-scoped** (inside a pipeline run): one shared resource per ``run_id``,
  created lazily on first use.  Access is serialized by a per-run lock
  (parallel DAG waves share one resource safely).  Finalization is deferred
  to :meth:`afinalize_run` — typically called from
  ``Service.ateardown(success=...)``.

The resource is keyed by ``run_id``, not by adapter instance, so a single
adapter shared across concurrent runs does not cross-contaminate
transactions.

This primitive lives in the database plugin (not the kernel): it only reads
``get_run_id`` from the kernel execution context (a one-way dependency) and is
consumed solely by transactional adapters.

Example — a dual-mode database adapter::

    class MyDbAdapter(Service):
        def __init__(self, session_factory) -> None:
            async def _open():
                return session_factory()

            async def _finalize(session, success):
                try:
                    if success:
                        await session.commit()
                    else:
                        await session.rollback()
                finally:
                    await session.close()

            self._scope = RunScopedResource(_open, _finalize)

        def get_session(self):
            return self._scope.aget()

        async def ateardown(self, *, success: bool = True) -> None:
            await self._scope.afinalize_run(success=success)

.. warning::
    Do not nest ``aget()`` calls within the same task while inside a run —
    the per-run lock is not re-entrant.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from hexdag.kernel.context.execution_context import get_run_id

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable


@dataclass
class _RunEntry:
    """Per-run bookkeeping: the shared resource, its lock, and failure flag."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    resource: Any = None
    created: bool = False
    failed: bool = False


class RunScopedResource[T]:
    """Lazily-created resource shared within a pipeline run.

    Parameters
    ----------
    factory :
        Async callable creating the resource (e.g. open a session and
        begin a transaction).
    finalize :
        Async callable ``(resource, success)`` releasing the resource —
        commit when ``success`` is True, rollback otherwise.  Always
        called exactly once per created resource.
    """

    def __init__(
        self,
        factory: Callable[[], Awaitable[T]],
        finalize: Callable[[T, bool], Awaitable[None]],
    ) -> None:
        self._factory = factory
        self._finalize = finalize
        self._per_run: dict[str, _RunEntry] = {}

    @asynccontextmanager
    async def aget(self) -> AsyncIterator[T]:
        """Yield the resource for the current scope.

        Standalone (no active run): fresh resource, finalized on exit.
        Run-scoped: the run's shared resource, finalized later by
        :meth:`afinalize_run`.  An exception raised inside the block marks
        the run as failed — finalization then rolls back even if
        ``afinalize_run(success=True)`` is called.
        """
        run_id = get_run_id()
        if run_id is None:
            resource = await self._factory()
            try:
                yield resource
            except BaseException:
                await self._finalize(resource, False)
                raise
            else:
                await self._finalize(resource, True)
        else:
            entry = self._per_run.setdefault(run_id, _RunEntry())
            async with entry.lock:
                if not entry.created:
                    entry.resource = await self._factory()
                    entry.created = True
                try:
                    yield entry.resource
                except BaseException:
                    entry.failed = True
                    raise

    def mark_failed(self) -> None:
        """Mark the current run's resource as failed (forces rollback)."""
        run_id = get_run_id()
        if run_id is not None and run_id in self._per_run:
            self._per_run[run_id].failed = True

    def mark_recovered(self) -> None:
        """Clear the current run's failure flag after a recoverable error.

        A step exception inside :meth:`aget` flips the run's ``failed`` flag,
        which would otherwise force a rollback of the whole run at
        :meth:`afinalize_run`.  When that failure was contained — e.g. rolled
        back to a SAVEPOINT so only one scope is undone while the run
        continues — call this to let the outer transaction commit normally.
        """
        run_id = get_run_id()
        if run_id is not None and run_id in self._per_run:
            self._per_run[run_id].failed = False

    async def afinalize_run(self, *, success: bool = True) -> None:
        """Finalize the current run's resource (commit/rollback + release).

        Commits only when ``success`` is True AND no step failed inside an
        ``aget()`` block.  No-op when the run never created the resource.
        Call from ``Service.ateardown(success=...)`` — or from ``aclose()``
        when mounted as a port (where no success flag exists, the internal
        failure flag is the only rollback signal).
        """
        run_id = get_run_id()
        if run_id is None:
            return
        entry = self._per_run.pop(run_id, None)
        if entry is None or not entry.created:
            return
        await self._finalize(entry.resource, success and not entry.failed)

    async def afinalize_all(self, *, success: bool = False) -> None:
        """Finalize every outstanding run resource (shutdown safety net).

        Defaults to rollback — orderly runs should have finalized via
        :meth:`afinalize_run` already; anything left is abnormal.
        """
        entries = [e for e in self._per_run.values() if e.created]
        self._per_run.clear()
        for entry in entries:
            await self._finalize(entry.resource, success and not entry.failed)
