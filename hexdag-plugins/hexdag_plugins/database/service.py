# type: ignore
"""DatabaseService — a ready-made, port-backed DB service for hexDAG.

Replaces the hand-rolled ``@db_step`` mixins applications otherwise write
(e.g. Raven's ``RavenDatabaseAdapter``).  It ships ``@tool``/``@step``
methods for the common CRUD + query operations and manages the pipeline-run
transaction declaratively via the dual-mode session pattern:

- **Standalone** (outside a run): every method autocommits independently.
- **Pipeline run**: all methods share one session; the run commits atomically
  at ``ateardown(success=True)`` or rolls back on any failure
  (``success=False`` / a step exception).  This is the whole-run-atomic
  default — the same guarantee Raven gets today, but generic.

For *partial* rollback, group a subset of steps with the ``transaction`` macro
(:mod:`hexdag_plugins.database.macros.transaction`): it brackets the steps with
SAVEPOINT ``begin`` / ``release`` nodes and routes step failures to a
``rollback`` node so only that scope is undone while the run continues.

Steps should ``flush()`` but never ``commit()`` — commit/rollback is owned by
the mode, not the step.
"""

from __future__ import annotations

from typing import Any

from hexdag.kernel.context.execution_context import get_run_id
from hexdag.kernel.exceptions import HexDAGError
from hexdag.kernel.service import Service, step, tool

from hexdag_plugins.database.dual_mode import DualModeSessionMixin

_IDENTIFIER_RE = r"^[A-Za-z_][A-Za-z0-9_]*$"


def _check_identifier(name: str, kind: str) -> str:
    """Validate a SQL identifier (table/column) to keep it out of injection paths."""
    import re

    if not re.match(_IDENTIFIER_RE, name):
        raise ValueError(f"Invalid {kind} identifier: {name!r}")
    return name


class DatabaseService(DualModeSessionMixin, Service):
    """SQLAlchemy-backed service exposing CRUD + query as tools/steps.

    Parameters
    ----------
    dsn : str | None
        Database connection string (e.g.
        ``postgresql+asyncpg://user:pass@host/db``).  Mutually exclusive with
        ``session_factory``.
    session_factory : Any | None
        Pre-built ``async_sessionmaker`` from the host application.  When
        provided, the service uses it directly and never builds its own
        engine — the recommended wiring for apps that already own a session
        factory (FastAPI, etc.).
    """

    def __init__(
        self,
        dsn: str | None = None,
        session_factory: Any | None = None,
    ) -> None:
        if dsn is None and session_factory is None:
            raise ValueError("DatabaseService requires either 'dsn' or 'session_factory'")
        self._dsn = dsn
        self._session_factory = session_factory
        self._engine = None
        # Per-(run, scope) SAVEPOINT handles for the transaction macro.
        self._savepoints: dict[tuple[str, str], Any] = {}
        self._init_dual_mode(self._make_session)

    def _make_session(self) -> Any:
        """Create a session from the configured or lazily-built factory."""
        if self._session_factory is None:
            from sqlalchemy.ext.asyncio import (
                AsyncSession,
                async_sessionmaker,
                create_async_engine,
            )

            self._engine = create_async_engine(self._dsn)
            self._session_factory = async_sessionmaker(
                self._engine, class_=AsyncSession, expire_on_commit=False
            )
        return self._session_factory()

    async def ateardown(self, *, success: bool = True) -> None:
        """Finalize the run transaction, then dispose any owned engine."""
        await super().ateardown(success=success)
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None

    # ------------------------------------------------------------------
    # Query / read
    # ------------------------------------------------------------------

    @tool
    @step
    async def query(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Run a parameterized SELECT and return rows as dicts."""
        from sqlalchemy import text

        async with self.get_session() as session:
            result = await session.execute(text(sql), params or {})
            return [dict(row._mapping) for row in result]

    @tool
    @step
    async def get(self, table: str, id: str, id_column: str = "id") -> dict[str, Any] | None:
        """Fetch a single row by primary key, or ``None`` if absent."""
        from sqlalchemy import text

        _check_identifier(table, "table")
        _check_identifier(id_column, "column")
        sql = f"SELECT * FROM {table} WHERE {id_column} = :id"  # nosec B608
        async with self.get_session() as session:
            result = await session.execute(text(sql), {"id": id})
            row = result.fetchone()
            return dict(row._mapping) if row is not None else None

    # ------------------------------------------------------------------
    # Write — staged with flush(), committed by the run/savepoint
    # ------------------------------------------------------------------

    @step
    async def insert(self, table: str, data: dict[str, Any]) -> dict[str, Any]:
        """Insert a row from ``data`` (column → value)."""
        from sqlalchemy import text

        _check_identifier(table, "table")
        cols = [_check_identifier(c, "column") for c in data]
        placeholders = ", ".join(f":{c}" for c in cols)
        col_list = ", ".join(cols)
        sql = f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})"  # nosec B608
        async with self.get_session() as session:
            await session.execute(text(sql), data)
            await session.flush()
        return {"inserted": True, "table": table, "data": data}

    @step
    async def update(
        self, table: str, id: str, data: dict[str, Any], id_column: str = "id"
    ) -> dict[str, Any]:
        """Update the row identified by ``id`` with ``data``."""
        from sqlalchemy import text

        _check_identifier(table, "table")
        _check_identifier(id_column, "column")
        cols = [_check_identifier(c, "column") for c in data]
        assignments = ", ".join(f"{c} = :{c}" for c in cols)
        sql = f"UPDATE {table} SET {assignments} WHERE {id_column} = :_id"  # nosec B608
        params = {**data, "_id": id}
        async with self.get_session() as session:
            result = await session.execute(text(sql), params)
            await session.flush()
        return {"updated": True, "table": table, "id": id, "rowcount": result.rowcount}

    @step
    async def delete(self, table: str, id: str, id_column: str = "id") -> dict[str, Any]:
        """Delete the row identified by ``id``."""
        from sqlalchemy import text

        _check_identifier(table, "table")
        _check_identifier(id_column, "column")
        sql = f"DELETE FROM {table} WHERE {id_column} = :id"  # nosec B608
        async with self.get_session() as session:
            result = await session.execute(text(sql), {"id": id})
            await session.flush()
        return {"deleted": True, "table": table, "id": id, "rowcount": result.rowcount}

    @step
    async def execute(self, sql: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute an arbitrary parameterized statement (no rows returned)."""
        from sqlalchemy import text

        async with self.get_session() as session:
            result = await session.execute(text(sql), params or {})
            await session.flush()
        return {"rowcount": getattr(result, "rowcount", -1)}

    # ------------------------------------------------------------------
    # SAVEPOINT control — used by the `transaction` macro
    # ------------------------------------------------------------------

    def _savepoint_key(self, scope_id: str) -> tuple[str, str]:
        return (get_run_id() or "_standalone", scope_id)

    @step
    async def begin_savepoint(self, scope_id: str) -> dict[str, Any]:
        """Open a nested SAVEPOINT for ``scope_id`` on the run's session.

        Only **one** ``transaction`` scope is supported per pipeline run.  Two
        scopes in one run share a single session, and rolling back one
        SAVEPOINT discards the other (savepoint-stack semantics), so a second
        open scope is rejected instead of silently corrupting the session.  For
        multiple independent atomic units, run them as separate processes in a
        ``kind: System`` — one process is one run is one transaction.
        """
        run_id = get_run_id() or "_standalone"
        if any(key[0] == run_id for key in self._savepoints):
            raise HexDAGError(
                "only one `transaction` scope is supported per pipeline run; "
                "for multiple independent atomic units, run them as separate "
                "processes in a `kind: System` (one process = one transaction)."
            )
        async with self.get_session() as session:
            self._savepoints[self._savepoint_key(scope_id)] = await session.begin_nested()
        return {"scope_id": scope_id, "savepoint": "open"}

    @step
    async def release_savepoint(self, scope_id: str) -> dict[str, Any]:
        """Release (commit) the SAVEPOINT — its writes fold into the run tx."""
        sp = self._savepoints.pop(self._savepoint_key(scope_id), None)
        if sp is not None and sp.is_active:
            await sp.commit()
        return {"scope_id": scope_id, "savepoint": "released"}

    @step
    async def rollback_savepoint(self, scope_id: str) -> dict[str, Any]:
        """Roll back the SAVEPOINT — undo only this scope; the run continues.

        A failing step inside the scope flips the run-level failure flag (which
        would force a full-run rollback at teardown).  Because the failure is
        now contained to this SAVEPOINT, clear that flag so the outer
        transaction still commits the rest of the run.
        """
        sp = self._savepoints.pop(self._savepoint_key(scope_id), None)
        if sp is not None and sp.is_active:
            await sp.rollback()
            # Only clear the run failure flag when a rollback actually happened —
            # otherwise a stray call could let a genuinely-failed run commit.
            self._db_scope.mark_recovered()
        return {"scope_id": scope_id, "savepoint": "rolled_back"}
