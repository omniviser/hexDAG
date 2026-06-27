# type: ignore
"""Dual-mode session management for SQLAlchemy-backed adapters.

Thin wrapper over :class:`~hexdag_plugins.database.run_scope.RunScopedResource`:

- **Standalone mode** (no active pipeline run): each ``get_session()``
  yields a fresh session, committed on success / rolled back on exception.
- **Pipeline mode** (inside a run): all ``get_session()`` calls share one
  session, serialized by a lock.  The transaction commits or rolls back
  once, in ``ateardown(success=...)`` — or in ``aclose()`` when the
  adapter is mounted as a port (no success flag; a step exception inside
  ``get_session()`` is the rollback signal).
"""

from collections.abc import Callable
from typing import Any

from hexdag_plugins.database.run_scope import RunScopedResource


class DualModeSessionMixin:
    """Mix into an adapter/service that owns an async session factory.

    Call :meth:`_init_dual_mode` with a zero-arg session factory (e.g. an
    ``async_sessionmaker``) before using :meth:`get_session`.
    """

    _db_scope: RunScopedResource[Any]

    def _init_dual_mode(self, session_factory: Callable[[], Any]) -> None:
        async def _open() -> Any:
            return session_factory()

        async def _finalize(session: Any, success: bool) -> None:
            try:
                if success:
                    await session.commit()
                else:
                    await session.rollback()
            finally:
                await session.close()

        self._db_scope = RunScopedResource(_open, _finalize)

    def get_session(self) -> Any:
        """Async context manager yielding the mode-appropriate session.

        Steps should ``flush()`` but not ``commit()`` — commit/rollback is
        owned by the mode (per-call in standalone, per-run in pipeline).
        """
        return self._db_scope.aget()

    async def ateardown(self, *, success: bool = True) -> None:
        """Finalize the run's shared session (Service mounting)."""
        await self._db_scope.afinalize_run(success=success)

    async def afinalize_sessions(self, *, success: bool = True) -> None:
        """Finalize the current run's session, then roll back any leftovers.

        Called from ``aclose()`` for port mounting, where no success flag
        exists: commits unless a step failed inside ``get_session()``.
        """
        await self._db_scope.afinalize_run(success=success)
        await self._db_scope.afinalize_all(success=False)
