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
from hexdag_plugins.database.dual_mode import DualModeSessionMixin


class SQLAlchemyAdapter(
    HexDAGAdapter,
    DualModeSessionMixin,
    SupportsQuery,
    SupportsRawSQL,
    SupportsSessionFactory,
    yaml_alias="sqlalchemy_adapter",
    port="database",
):
    """Adapter for SQLAlchemy-supported databases.

    Session access comes in two flavors:

    - ``asession()`` — always-independent session (saga mode: each caller
      commits its own work, see ``SupportsSessionFactory``).
    - ``get_session()`` — dual-mode session: per-call autocommit outside a
      pipeline run, one shared atomic transaction inside a run (commit or
      rollback decided at ``ateardown(success=...)`` / ``aclose()``).
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
        self._init_dual_mode(self._make_session)

    async def connect(self) -> None:
        """Establish database connection."""
        self._ensure_engine()
        async with self.engine.connect() as conn:
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

    async def aget_table_schemas(self) -> dict[str, dict[str, Any]]:
        """
        Get schema information for all tables in Database format.

        Returns:
            Dictionary mapping table names to schema information.
        """
        if not self.engine:
            raise RuntimeError("Not connected to database")

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
        """Execute a SQL query and return results.

        Args:
            query: SQL query string
            params: Optional query parameters for safe parameterized queries

        Returns:
            List of dictionaries representing query result rows
        """
        if not self.engine:
            raise RuntimeError("Not connected to database")

        results = []
        async with self.engine.connect() as conn:
            result = await conn.execute(text(query), params or {})
            results.extend(dict(row._mapping) for row in result)
        return results

    async def get_table_schemas(self) -> Sequence[TableSchema]:
        """
        Get schema information for all tables.

        Returns:
            Sequence[TableSchema]: List of table schemas
        """
        if not self.engine:
            raise RuntimeError("Not connected to database")

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

        Args:
            table: Table name
            filters: Optional column-value pairs to filter by
            columns: Optional list of columns to return (None = all)
            limit: Optional maximum number of rows to return
        Returns:
            AsyncIterator[dict[str, Any]]: An async iterator over rows as dictionaries
        """
        if not self.engine:
            raise RuntimeError("Not connected to database")

        table_obj = Table(table, self._metadata, extend_existing=True)
        query = select(table_obj)

        if columns:
            query = select(*[table_obj.c[col] for col in columns])

        if filters:
            conditions = [table_obj.c[k] == v for k, v in filters.items()]
            query = query.where(*conditions)

        if limit:
            query = query.limit(limit)

        async def generate_rows() -> AsyncIterator[dict[str, Any]]:
            if not self.engine:  # Recheck engine in case it was closed
                raise RuntimeError("Database connection lost")
            async with self.engine.connect() as conn:
                result = await conn.stream(query)
                async for row in result:
                    yield dict(row._mapping)

        return generate_rows()

    def query_raw(
        self, sql: str, params: dict[str, Any] | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Execute a raw SQL query."""
        if not self.engine:
            raise RuntimeError("Not connected to database")

        async def generate_rows() -> AsyncIterator[dict[str, Any]]:
            if not self.engine:
                raise RuntimeError("Database connection lost")
            async with self.engine.connect() as conn:
                result = await conn.stream(text(sql), params or {})
                async for row in result:
                    yield dict(row._mapping)

        return generate_rows()

    async def execute_raw(self, sql: str, params: dict[str, Any] | None = None) -> None:
        """
        Execute a raw SQL statement without returning results.

        Args:
            sql: SQL statement to execute
            params: Optional parameters for the SQL statement
        """
        if not self.engine:
            raise RuntimeError("Not connected to database")

        async with self.engine.connect() as conn:
            await conn.execute(text(sql), params or {})
            await conn.commit()

    async def get_indexes(self, table: str) -> list[str]:
        """
        Get index information for a table.

        Args:
            table: Table name

        Returns:
            list[str]: List of index names
        """
        if not self.engine:
            raise RuntimeError("Not connected to database")

        async with self.engine.connect() as conn:
            # Remove engine argument from inspect call
            inspector = await conn.run_sync(inspect)
            # Filter out None values to ensure list[str]
            return [idx["name"] for idx in inspector.get_indexes(table) if idx["name"] is not None]

    async def get_table_statistics(self, table: str) -> dict[str, int]:
        """
        Get statistical information about a table.

        Args:
            table: Table name

        Returns:
            dict[str, int]: Statistics including row count
        """
        if not self.engine:
            raise RuntimeError("Not connected to database")

        async with self.engine.connect() as conn:
            result = await conn.execute(text(f"SELECT COUNT(*) as count FROM {table}"))  # nosec
            row = await result.fetchone()  # type: ignore # type: ignore
            return {"row_count": int(row[0]) if row else 0}
