# type: ignore
"""SQLAlchemy adapter implementation for hexDAG."""

from collections.abc import AsyncIterator, Sequence
from typing import Any

from sqlalchemy import MetaData, Table, inspect, select, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from hexai.ports.database import (
    ColumnSchema,
    DatabasePort,
    SupportsIndexes,
    SupportsRawSQL,
    SupportsStatistics,
    TableSchema,
)


class SQLAlchemyAdapter(DatabasePort, SupportsRawSQL, SupportsIndexes, SupportsStatistics):
    """Adapter for SQLAlchemy-supported databases."""

    def __init__(self, dsn: str) -> None:
        """
        Initialize SQLAlchemy adapter.

        Args:
            dsn: Database connection string (e.g., postgresql+asyncpg://user:pass@localhost/db)
        """
        self.engine: AsyncEngine | None = None
        self.dsn = dsn
        self._metadata = MetaData()

    async def connect(self) -> None:
        """Establish database connection."""
        self.engine = create_async_engine(self.dsn)
        async with self.engine.connect() as conn:
            # Force reflection after any schema changes
            await conn.run_sync(self._metadata.reflect)

    async def disconnect(self) -> None:
        """Close database connection."""
        if self.engine:
            await self.engine.dispose()
            self.engine = None

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

        Yields:
            dict[str, Any]: Each row as a dictionary
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
                    # Convert SQLAlchemy Row to dict using _mapping
                    yield {key: value for key, value in row._mapping.items()}

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
                    # Convert SQLAlchemy Row to dict properly
                    yield {key: value for key, value in row._mapping.items()}

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
