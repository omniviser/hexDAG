"""Run-scoped sessions — one shared transaction per pipeline run.

A :class:`RunScopedSessions` answers "give me the session scoped to this
pipeline run, or a standalone one if I'm outside a pipeline" — the pattern
every transactional adapter otherwise reimplements by hand (shared session,
concurrency lock, failure flag, commit-or-rollback decision).

The scope is selected implicitly by the execution context:

- **Standalone** (no active run): each :meth:`aget` creates a fresh session
  and finalizes it immediately on exit — commit on success, rollback on
  exception.
- **Run-scoped** (inside a pipeline run): one shared session per ``run_id``,
  created lazily on first use.  Access is serialized by a per-run lock
  (parallel DAG waves share one session safely).  Nothing commits mid-run;
  finalization is deferred to :meth:`afinalize_run` — typically called from
  ``ateardown(success=...)``.

Sessions are keyed by ``run_id``, not by adapter instance, so a single
adapter shared across concurrent runs does not cross-contaminate
transactions.

This lives in the database plugin (not the kernel): it only reads
``get_run_id`` from the kernel execution context (a one-way dependency) and is
consumed by transactional adapters/services via composition::

    class MyDbAdapter(Service):
        def __init__(self, session_factory) -> None:
            self._sessions = RunScopedSessions(session_factory)

        def get_session(self):
            return self._sessions.aget()

        async def ateardown(self, *, success: bool = True) -> None:
            await self._sessions.afinalize_run(success=success)

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
    from collections.abc import AsyncIterator, Callable


@dataclass
class _RunEntry:
    """Per-run bookkeeping: the shared session, its lock, and failure flag."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    session: Any = None
    created: bool = False
    failed: bool = False


class RunScopedSessions:
    """One shared session per pipeline run; fresh autocommitting sessions outside.

    Parameters
    ----------
    session_factory :
        Zero-arg callable creating a session (e.g. an ``async_sessionmaker``).

    Finalization policy: commit on success, rollback otherwise, always close.
    Applied per-call in standalone mode, once per run at :meth:`afinalize_run`
    in run-scoped mode.
    """

    def __init__(self, session_factory: Callable[[], Any]) -> None:
        self._session_factory = session_factory
        self._per_run: dict[str, _RunEntry] = {}

    async def _finalize(self, session: Any, success: bool) -> None:
        try:
            if success:
                await session.commit()
            else:
                await session.rollback()
        finally:
            await session.close()

    @asynccontextmanager
    async def aget(self) -> AsyncIterator[Any]:
        """Yield the session for the current scope.

        Standalone (no active run): fresh session, committed/rolled back and
        closed on exit.  Run-scoped: the run's shared session, finalized later
        by :meth:`afinalize_run`.  An exception raised inside the block marks
        the run as failed — finalization then rolls back even if
        ``afinalize_run(success=True)`` is called.
        """
        run_id = get_run_id()
        if run_id is None:
            session = self._session_factory()
            try:
                yield session
            except BaseException:
                await self._finalize(session, False)
                raise
            else:
                await self._finalize(session, True)
        else:
            entry = self._per_run.setdefault(run_id, _RunEntry())
            async with entry.lock:
                if not entry.created:
                    entry.session = self._session_factory()
                    entry.created = True
                try:
                    yield entry.session
                except BaseException:
                    entry.failed = True
                    raise

    def mark_failed(self) -> None:
        """Mark the current run's session as failed (forces rollback)."""
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
        """Finalize the current run's session (commit/rollback + close).

        Commits only when ``success`` is True AND no step failed inside an
        ``aget()`` block.  No-op when the run never created the session.
        Call from ``ateardown(success=...)`` — or from ``aclose()`` when
        mounted as a port (where no success flag exists, the internal
        failure flag is the only rollback signal).
        """
        run_id = get_run_id()
        if run_id is None:
            return
        entry = self._per_run.pop(run_id, None)
        if entry is None or not entry.created:
            return
        await self._finalize(entry.session, success and not entry.failed)

    async def afinalize_all(self, *, success: bool = False) -> None:
        """Finalize every outstanding run session (shutdown safety net).

        Defaults to rollback — orderly runs should have finalized via
        :meth:`afinalize_run` already; anything left is abnormal.
        """
        entries = [e for e in self._per_run.values() if e.created]
        self._per_run.clear()
        for entry in entries:
            await self._finalize(entry.session, success and not entry.failed)
