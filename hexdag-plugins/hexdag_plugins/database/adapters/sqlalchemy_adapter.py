# type: ignore
"""SQLAlchemy adapter implementation for hexDAG (database plugin)."""

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

from hexdag.kernel.ports.data_store import SupportsQuery
from hexdag.kernel.ports.database import (
    ColumnSchema,
    SupportsRawSQL,
    TableSchema,
)
from hexdag.stdlib.adapters.base import HexDAGAdapter
from sqlalchemy import MetaData, Table, inspect, select, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from hexdag_plugins.database._ports import SupportsSessionFactory
from hexdag_plugins.database.run_scope import RunScopedSessions


class SQLAlchemyAdapter(
    HexDAGAdapter,
    SupportsQuery,
    SupportsRawSQL,
    SupportsSessionFactory,
    yaml_alias="sqlalchemy_adapter",
    port="database",
):
    """Adapter for SQLAlchemy-supported databases.

    All query/execute methods are transaction-aware: they run on the
    run-scoped session (see :class:`RunScopedSessions`), so inside a
    pipeline run they see the run's uncommitted writes and their own
    writes join the run transaction.  Session access comes in two flavors:

    - ``get_session()`` — run-scoped session: per-call autocommit outside a
      pipeline run, one shared atomic transaction inside a run (commit or
      rollback decided at ``ateardown(success=...)`` / ``aclose()``).
    - ``asession()`` — always-independent session, out-of-band by design
      (saga mode: each caller commits its own work, see
      ``SupportsSessionFactory``).
    """

    def __init__(
        self,
        dsn: str | None = None,
        session_factory: Any | None = None,
    ) -> None:
        """
        Initialize SQLAlchemy adapter.

        Args:
            dsn: Database connection string (e.g., postgresql+asyncpg://user:pass@localhost/db)
            session_factory: Pre-built ``async_sessionmaker`` from the host
                app.  When provided, the adapter uses it instead of creating
                an engine from ``dsn`` and never disposes the host's engine.
        """
        if dsn is None and session_factory is None:
            msg = "SQLAlchemyAdapter requires either 'dsn' or 'session_factory'"
            raise ValueError(msg)
        self.engine: AsyncEngine | None = None
        self.dsn = dsn
        self._metadata = MetaData()
        self._session_factory = session_factory
        self._owns_engine = session_factory is None
        self._sessions = RunScopedSessions(self._make_session)

    async def connect(self) -> None:
        """Reflect the database schema into ``self._metadata``."""
        async with self.get_session() as session:
            conn = await session.connection()
            # Force reflection after any schema changes
            await conn.run_sync(self._metadata.reflect)

    def _ensure_engine(self) -> None:
        """Create the engine lazily from the DSN."""
        if self.engine is None:
            if self.dsn is None:
                msg = "Not connected to database (no DSN; session_factory-only adapter)"
                raise RuntimeError(msg)
            self.engine = create_async_engine(self.dsn)

    def _make_session(self) -> AsyncSession:
        """Create a session from the configured or lazily-built factory."""
        if self._session_factory is None:
            self._ensure_engine()
            self._session_factory = async_sessionmaker(
                self.engine, class_=AsyncSession, expire_on_commit=False
            )
        return self._session_factory()

    def get_session(self) -> Any:
        """Async context manager yielding the run-scoped session.

        Steps should ``flush()`` but not ``commit()`` — commit/rollback is
        owned by the scope (per-call standalone, per-run in a pipeline).
        """
        return self._sessions.aget()

    async def ateardown(self, *, success: bool = True) -> None:
        """Finalize the run's shared session (Service mounting)."""
        await self._sessions.afinalize_run(success=success)

    async def afinalize_sessions(self, *, success: bool = True) -> None:
        """Finalize the current run's session, then roll back any leftovers.

        Called from ``aclose()`` for port mounting, where no success flag
        exists: commits unless a step failed inside ``get_session()``.
        """
        await self._sessions.afinalize_run(success=success)
        await self._sessions.afinalize_all(success=False)

    @asynccontextmanager
    async def asession(self) -> AsyncIterator[AsyncSession]:
        """Independent session with its own transaction scope (saga mode).

        The caller commits or rolls back; the connection returns to the
        pool on exit.  See ``SupportsSessionFactory``.
        """
        session = self._make_session()
        try:
            yield session
        finally:
            await session.close()

    async def aclose(self) -> None:
        """Finalize outstanding sessions and close the database connection."""
        await self.afinalize_sessions(success=True)
        # self.engine is always adapter-created (hosts inject a factory,
        # never an engine), so disposing it never touches host resources.
        if self.engine:
            await self.engine.dispose()
            self.engine = None
        if self._owns_engine:
            # The lazily-built factory is bound to the disposed engine and
            # would otherwise hand out broken sessions after a reconnect.
            self._session_factory = None

    # Backward-compatible alias
    disconnect = aclose

    async def _fetch_all(
        self, stmt: Any, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute ``stmt`` on the run-scoped session and materialize all rows."""
        async with self.get_session() as session:
            result = await session.execute(stmt, params or {})
            return [dict(row._mapping) for row in result]

    async def aget_table_schemas(self) -> dict[str, dict[str, Any]]:
        """
        Get schema information for all tables in Database format.

        Returns:
            Dictionary mapping table names to schema information.
        """
        schemas = {}
        for table_name, table in self._metadata.tables.items():
            columns = {}
            primary_keys = []
            foreign_keys = []

            for col in table.columns:
                columns[col.name] = str(col.type)

                if col.primary_key:
                    primary_keys.append(col.name)

                if col.foreign_keys:
                    fk = next(iter(col.foreign_keys))
                    foreign_keys.append(
                        {
                            "from_column": col.name,
                            "to_table": fk.column.table.name,
                            "to_column": fk.column.name,
                        }
                    )

            schemas[table_name] = {
                "table_name": table_name,
                "columns": columns,
                "primary_keys": primary_keys,
                "foreign_keys": foreign_keys,
            }

        return schemas

    async def aexecute_query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a SQL query on the run-scoped session and return results.

        Args:
            query: SQL query string
            params: Optional query parameters for safe parameterized queries

        Returns:
            List of dictionaries representing query result rows
        """
        return await self._fetch_all(text(query), params)

    async def get_table_schemas(self) -> Sequence[TableSchema]:
        """
        Get schema information for all tables.

        Returns:
            Sequence[TableSchema]: List of table schemas
        """
        schemas = []
        for table_name, table in self._metadata.tables.items():
            columns = []
            for col in table.columns:
                foreign_key = None
                if col.foreign_keys:
                    fk = next(iter(col.foreign_keys))
                    foreign_key = f"{fk.column.table.name}.{fk.column.name}"

                nullable = bool(col.nullable) if col.nullable is not None else False

                columns.append(
                    ColumnSchema(
                        name=col.name,
                        type=col.type.value,
                        nullable=nullable,
                        primary_key=col.primary_key,
                        foreign_key=foreign_key,
                    )
                )
            schemas.append(TableSchema(name=table_name, columns=columns))

        return schemas

    def query(
        self,
        table: str,
        filters: dict[str, Any] | None = None,
        columns: list[str] | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Query rows from a table with filtering and column selection.

        Results are fully materialized before yielding (the run-scoped
        session lock is held only while the query executes).  For true
        streaming of large result sets use ``asession()`` with
        ``session.stream()``.

        Args:
            table: Table name
            filters: Optional column-value pairs to filter by
            columns: Optional list of columns to return (None = all)
            limit: Optional maximum number of rows to return
        Returns:
            AsyncIterator[dict[str, Any]]: An async iterator over rows as dictionaries
        """
        table_obj = Table(table, self._metadata, extend_existing=True)
        stmt = select(table_obj)

        if columns:
            stmt = select(*[table_obj.c[col] for col in columns])

        if filters:
            conditions = [table_obj.c[k] == v for k, v in filters.items()]
            stmt = stmt.where(*conditions)

        if limit:
            stmt = stmt.limit(limit)

        async def generate_rows() -> AsyncIterator[dict[str, Any]]:
            for row in await self._fetch_all(stmt):
                yield row

        return generate_rows()

    def query_raw(
        self, sql: str, params: dict[str, Any] | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Execute a raw SQL query on the run-scoped session.

        Results are fully materialized before yielding — see :meth:`query`.
        """

        async def generate_rows() -> AsyncIterator[dict[str, Any]]:
            for row in await self._fetch_all(text(sql), params):
                yield row

        return generate_rows()

    async def execute_raw(self, sql: str, params: dict[str, Any] | None = None) -> None:
        """
        Execute a raw SQL statement without returning results.

        Runs on the run-scoped session: standalone calls autocommit, calls
        inside a pipeline run join the run transaction (committed or rolled
        back at teardown).

        Args:
            sql: SQL statement to execute
            params: Optional parameters for the SQL statement
        """
        async with self.get_session() as session:
            await session.execute(text(sql), params or {})
            await session.flush()

    async def get_indexes(self, table: str) -> list[str]:
        """
        Get index information for a table.

        Args:
            table: Table name

        Returns:
            list[str]: List of index names
        """
        async with self.get_session() as session:
            conn = await session.connection()
            indexes = await conn.run_sync(lambda sync_conn: inspect(sync_conn).get_indexes(table))
        return [idx["name"] for idx in indexes if idx["name"] is not None]

    async def get_table_statistics(self, table: str) -> dict[str, int]:
        """
        Get statistical information about a table.

        Args:
            table: Table name

        Returns:
            dict[str, int]: Statistics including row count
        """
        rows = await self._fetch_all(text(f"SELECT COUNT(*) as count FROM {table}"))  # nosec
        return {"row_count": int(rows[0]["count"]) if rows else 0}
