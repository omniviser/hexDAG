"""SQLite database adapter implementation with async support."""

import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiosqlite

from hexdag.core.logging import get_logger
from hexdag.core.registry.decorators import adapter
from hexdag.core.utils.sql_validation import validate_sql_identifier

logger = get_logger(__name__)


@adapter(
    name="sqlite",
    implements_port="database",
    description="Async SQLite database adapter for SQL query execution and schema introspection",
)
class SQLiteAdapter:
    """Async SQLite adapter for database port.

    Provides a lightweight, file-based database solution that implements
    the DatabasePort interface for SQL execution and schema introspection.
    All operations are fully async using aiosqlite.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize SQLite adapter.

        Args
        ----
            **kwargs: Configuration options (db_path, timeout, etc.)
        """
        db_path = kwargs.get("db_path", ":memory:")
        self.db_path = Path(db_path) if db_path != ":memory:" else db_path
        self.check_same_thread = kwargs.get("check_same_thread", False)
        self.timeout = kwargs.get("timeout", 5.0)
        self.journal_mode = kwargs.get("journal_mode", "WAL")
        self.foreign_keys = kwargs.get("foreign_keys", True)
        self.read_only = kwargs.get("read_only", False)

        self.connection: aiosqlite.Connection | None = None

    async def _ensure_database(self) -> None:
        """Ensure database connection exists and is configured."""
        if self.connection is not None:
            return

        db_path = str(self.db_path) if isinstance(self.db_path, Path) else self.db_path

        if self.read_only and db_path != ":memory:":
            # Use URI mode for read-only access
            db_uri = f"file:{db_path}?mode=ro"
            self.connection = await aiosqlite.connect(
                db_uri,
                uri=True,
                check_same_thread=self.check_same_thread,
                timeout=self.timeout,
            )
        else:
            self.connection = await aiosqlite.connect(
                db_path,
                check_same_thread=self.check_same_thread,
                timeout=self.timeout,
            )

        self.connection.row_factory = aiosqlite.Row  # Return rows as dictionaries

        # Configure database settings
        if self.connection:
            async with self.connection.cursor() as cursor:
                if self.journal_mode:
                    await cursor.execute(f"PRAGMA journal_mode = {self.journal_mode}")
                if self.foreign_keys:
                    await cursor.execute("PRAGMA foreign_keys = ON")
            await self.connection.commit()

    @asynccontextmanager
    async def _get_cursor(self) -> AsyncIterator[aiosqlite.Cursor]:
        """Context manager for database cursor.

        Ensures proper cursor cleanup and error handling.

        Yields
        ------
        aiosqlite.Cursor
            Async SQLite database cursor

        Raises
        ------
        RuntimeError
            If database connection is not established
        aiosqlite.Error
            If database error occurs during operation
        """
        await self._ensure_database()
        if self.connection is None:
            raise RuntimeError("Database connection not established")

        async with self.connection.cursor() as cursor:
            try:
                yield cursor
            except aiosqlite.Error as e:
                logger.error(f"Database error: {e}")
                await self.connection.rollback()
                raise

    async def _get_all_tables(self) -> list[str]:
        """Get all non-system tables from the database.

        Returns
        -------
            List of table names
        """
        async with self._get_cursor() as cursor:
            await cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """)
            rows = await cursor.fetchall()
            return [row[0] for row in rows]

    def _validate_identifier(self, identifier: str, identifier_type: str = "table") -> bool:
        """Validate a database identifier to prevent injection.

        Parameters
        ----------
        identifier : str
            The identifier to validate (table name, column name, etc.)
        identifier_type : str
            Type of identifier for error messages

        Returns
        -------
        bool
            True if valid, False otherwise
        """
        return validate_sql_identifier(identifier, identifier_type=identifier_type)

    async def aget_table_schemas(self) -> dict[str, dict[str, Any]]:
        """Get schema information for all tables.

        Returns
        -------
            Dictionary mapping table names to schema information
        """
        tables = await self._get_all_tables()
        schemas = {}

        async with self._get_cursor() as cursor:
            for table in tables:
                if not self._validate_identifier(table):
                    continue

                await cursor.execute(f'PRAGMA table_info("{table}")')  # nosec B608 - validated
                rows = await cursor.fetchall()

                columns = {}
                primary_keys = []
                for row in rows:
                    col_name = row[1]
                    col_type = row[2]
                    is_pk = row[5]

                    columns[col_name] = col_type
                    if is_pk:
                        primary_keys.append(col_name)

                # nosec B608 - validated
                await cursor.execute(f'PRAGMA foreign_key_list("{table}")')
                fk_rows = await cursor.fetchall()
                foreign_keys = [
                    {"from_column": fk_row[3], "to_table": fk_row[2], "to_column": fk_row[4]}
                    for fk_row in fk_rows
                ]

                schemas[table] = {
                    "table_name": table,
                    "columns": columns,
                    "primary_keys": primary_keys,
                    "foreign_keys": foreign_keys,
                }

        return schemas

    async def aexecute_query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a SQL query and return results.

        Args
        ----
            query: SQL query to execute
            params: Optional query parameters for safe parameterized queries

        Returns
        -------
            List of dictionaries representing query result rows.
            For non-SELECT queries (INSERT/UPDATE/DELETE), returns an empty list
            and commits the transaction.

        Notes
        -----
        - SELECT/PRAGMA/WITH queries return result rows
        - INSERT/UPDATE/DELETE queries are committed automatically and return []
        - Use parameters to prevent SQL injection with :name placeholders

        Raises
        ------
        ValueError
            If a required parameter is missing
        aiosqlite.Error
            If query execution fails
        """
        try:
            async with self._get_cursor() as cursor:
                if params:
                    # SQLite uses ? placeholders, but we support :name format
                    param_names = []

                    def replacer(match: re.Match[str]) -> str:
                        param_names.append(match.group(1))
                        return "?"

                    converted_query = re.sub(r":(\w+)", replacer, query)

                    try:
                        param_values = [params[name] for name in param_names]
                        await cursor.execute(converted_query, param_values)
                    except KeyError as e:
                        raise ValueError(f"Missing parameter: {e}") from e
                else:
                    await cursor.execute(query)

                # Determine query type
                query_type = query.strip().upper().split()[0] if query.strip() else ""

                # For SELECT queries, fetch results
                if query_type in ("SELECT", "PRAGMA", "WITH"):
                    rows = await cursor.fetchall()
                    return [dict(row) for row in rows]
                # For INSERT/UPDATE/DELETE operations
                # Note: If database is opened in read-only mode, SQLite will
                # automatically raise an OperationalError for write operations
                affected_rows = cursor.rowcount
                if self.connection is not None:
                    await self.connection.commit()

                # Log the operation
                logger.info(f"Executed {query_type} query, affected rows: {affected_rows}")
                return []
        except aiosqlite.Error as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query: {query[:100]}...")  # Log first 100 chars of query
            raise

    async def aget_relationships(self) -> list[dict[str, Any]]:
        """Get foreign key relationships between tables.

        Returns
        -------
            List of relationship dictionaries
        """
        tables = await self._get_all_tables()
        relationships: list[dict[str, Any]] = []

        async with self._get_cursor() as cursor:
            for table in tables:
                if not self._validate_identifier(table):
                    continue

                # nosec B608 - validated
                await cursor.execute(f'PRAGMA foreign_key_list("{table}")')
                fk_rows = await cursor.fetchall()

                relationships.extend(
                    {
                        "from_table": table,
                        "from_column": fk_row[3],
                        "to_table": fk_row[2],
                        "to_column": fk_row[4],
                        "relationship_type": "many_to_one",  # SQLite doesn't store this info
                    }
                    for fk_row in fk_rows
                )

        return relationships

    async def aget_indexes(self) -> list[dict[str, Any]]:
        """Get index information for performance optimization.

        Returns
        -------
            List of index dictionaries
        """
        indexes = []

        async with self._get_cursor() as cursor:
            await cursor.execute("""
                SELECT name, tbl_name, sql FROM sqlite_master
                WHERE type='index' AND sql IS NOT NULL
            """)
            index_rows = await cursor.fetchall()

            for row in index_rows:
                index_name = row[0]
                table_name = row[1]

                if not self._validate_identifier(index_name, "index"):
                    continue
                if not self._validate_identifier(table_name, "table"):
                    continue

                await cursor.execute(f'PRAGMA index_info("{index_name}")')  # nosec B608 - validated
                col_rows = await cursor.fetchall()
                columns = [col_row[2] for col_row in col_rows]

                await cursor.execute(f'PRAGMA index_list("{table_name}")')  # nosec B608 - validated
                idx_list = await cursor.fetchall()
                is_unique = False
                for idx in idx_list:
                    if idx[1] == index_name:
                        is_unique = bool(idx[2])
                        break

                indexes.append({
                    "index_name": index_name,
                    "table_name": table_name,
                    "columns": columns,
                    "index_type": "btree",  # SQLite primarily uses B-tree
                    "is_unique": is_unique,
                })

        return indexes

    async def aget_table_statistics(self) -> dict[str, dict[str, Any]]:
        """Get table statistics for query optimization.

        Returns
        -------
            Dictionary mapping table names to statistics
        """
        tables = await self._get_all_tables()
        stats = {}

        async with self._get_cursor() as cursor:
            for table in tables:
                if not self._validate_identifier(table):
                    continue  # Skip invalid table names

                # SQLite uses double quotes for identifiers
                await cursor.execute(f'SELECT COUNT(*) FROM "{table}"')  # nosec B608 - validated
                result = await cursor.fetchone()
                row_count = result[0] if result else 0

                await cursor.execute(
                    "SELECT SUM(LENGTH(sql)) FROM sqlite_master WHERE tbl_name = ?", (table,)
                )
                size_result = await cursor.fetchone()
                size_bytes = size_result[0] if size_result and size_result[0] else 0

                stats[table] = {
                    "row_count": row_count,
                    "size_bytes": size_bytes,
                    "last_updated": None,  # SQLite doesn't track this
                }

        return stats

    async def close(self) -> None:
        """Close database connection."""
        if self.connection:
            await self.connection.close()
            self.connection = None

    async def __aenter__(self) -> "SQLiteAdapter":
        """Async context manager entry."""
        await self._ensure_database()
        return self

    async def __aexit__(
        self,
        _exc_type: Any,  # noqa: ARG002
        _exc_val: Any,  # noqa: ARG002
        _exc_tb: Any,  # noqa: ARG002
    ) -> None:
        """Async context manager exit."""
        await self.close()

    def __repr__(self) -> str:
        """Return string representation."""
        mode = "read-only" if self.read_only else "read-write"
        return f"SQLiteAdapter(db_path='{self.db_path}', mode='{mode}')"
