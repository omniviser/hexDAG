"""SQLite database adapter implementation."""

import logging
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from sqlite3 import Connection, Cursor
from typing import Any

from pydantic import BaseModel, Field

from hexai.core.ports.configurable import ConfigurableComponent
from hexai.core.registry.decorators import adapter

logger = logging.getLogger(__name__)


@adapter(
    name="sqlite",
    implements_port="database",
    description="SQLite database adapter for SQL query execution and schema introspection",
)
class SQLiteAdapter(ConfigurableComponent):
    """SQLite adapter for database port.

    Provides a lightweight, file-based database solution that implements
    the DatabasePort interface for SQL execution and schema introspection.
    """

    # Configuration schema for TOML generation
    class Config(BaseModel):
        """Configuration schema for SQLite adapter."""

        db_path: str = Field(
            default="hexdag.db",
            description="Path to SQLite database file (use ':memory:' for in-memory)",
        )
        check_same_thread: bool = Field(
            default=False,
            description="SQLite check_same_thread parameter (False for async compatibility)",
        )
        timeout: float = Field(default=5.0, gt=0, description="Connection timeout in seconds")
        journal_mode: str = Field(
            default="WAL", description="Journal mode (WAL for better concurrency)"
        )
        foreign_keys: bool = Field(default=True, description="Enable foreign key constraints")
        read_only: bool = Field(
            default=False, description="Open database in read-only mode (prevents modifications)"
        )

    @classmethod
    def get_config_class(cls) -> type[BaseModel]:
        """Return configuration schema."""
        return cls.Config

    def __init__(self, **kwargs: Any) -> None:
        """Initialize SQLite adapter.

        Args
        ----
            **kwargs: Configuration options (db_path, timeout, etc.)
        """
        # Create config from kwargs using the Config schema
        config_data = {}
        for field_name in self.Config.model_fields:
            if field_name in kwargs:
                config_data[field_name] = kwargs[field_name]

        # Create and validate config
        config = self.Config(**config_data)

        # Store configuration
        self.config = config
        self.db_path = Path(config.db_path) if config.db_path != ":memory:" else config.db_path
        self.check_same_thread = config.check_same_thread
        self.timeout = config.timeout
        self.journal_mode = config.journal_mode
        self.foreign_keys = config.foreign_keys
        self.read_only = config.read_only

        self.connection: Connection | None = None
        self._ensure_database()

    def _ensure_database(self) -> None:
        """Ensure database connection exists."""
        db_path = str(self.db_path) if isinstance(self.db_path, Path) else self.db_path

        # Add read-only mode support via URI
        if self.read_only and db_path != ":memory:":
            # Use URI mode for read-only access
            db_uri = f"file:{db_path}?mode=ro"
            self.connection = sqlite3.connect(
                db_uri,
                uri=True,
                check_same_thread=self.check_same_thread,
                timeout=self.timeout,
            )
        else:
            self.connection = sqlite3.connect(
                db_path,
                check_same_thread=self.check_same_thread,
                timeout=self.timeout,
            )
        self.connection.row_factory = sqlite3.Row  # Return rows as dictionaries

        # Configure database settings
        if self.connection:
            with self._get_cursor() as cursor:
                if self.journal_mode:
                    cursor.execute(f"PRAGMA journal_mode = {self.journal_mode}")
                if self.foreign_keys:
                    cursor.execute("PRAGMA foreign_keys = ON")

    @contextmanager
    def _get_cursor(self) -> Iterator[Cursor]:
        """Context manager for database cursor.

        Ensures proper cursor cleanup and error handling.
        """
        if self.connection is None:
            raise RuntimeError("Database connection not established")

        cursor = self.connection.cursor()
        try:
            yield cursor
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            self.connection.rollback()
            raise
        finally:
            cursor.close()

    def _get_all_tables(self) -> list[str]:
        """Get all non-system tables from the database.

        Returns
        -------
            List of table names
        """
        with self._get_cursor() as cursor:
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """)
            return [row[0] for row in cursor.fetchall()]

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
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", identifier):
            logger.warning(f"Invalid {identifier_type} name: {identifier}")
            return False
        return True

    async def aget_table_schemas(self) -> dict[str, dict[str, Any]]:
        """Get schema information for all tables.

        Returns
        -------
            Dictionary mapping table names to schema information
        """
        tables = self._get_all_tables()
        schemas = {}

        with self._get_cursor() as cursor:
            for table in tables:
                if not self._validate_identifier(table):
                    continue

                # Get column info
                cursor.execute(f'PRAGMA table_info("{table}")')  # nosec B608 - validated
                rows = cursor.fetchall()

                columns = {}
                primary_keys = []
                for row in rows:
                    col_name = row[1]
                    col_type = row[2]
                    is_pk = row[5]

                    columns[col_name] = col_type
                    if is_pk:
                        primary_keys.append(col_name)

                # Get foreign keys
                cursor.execute(f'PRAGMA foreign_key_list("{table}")')  # nosec B608 - validated
                fk_rows = cursor.fetchall()
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
        """
        try:
            with self._get_cursor() as cursor:
                # Handle parameterized queries
                if params:
                    # SQLite uses ? placeholders, but we support :name format
                    # Convert :name placeholders to ? for SQLite
                    param_names = []

                    def replacer(match: re.Match[str]) -> str:
                        param_names.append(match.group(1))
                        return "?"

                    converted_query = re.sub(r":(\w+)", replacer, query)

                    # Build parameter values in the correct order
                    try:
                        param_values = [params[name] for name in param_names]
                        cursor.execute(converted_query, param_values)
                    except KeyError as e:
                        raise ValueError(f"Missing parameter: {e}") from e
                else:
                    cursor.execute(query)

                # Determine query type
                query_type = query.strip().upper().split()[0] if query.strip() else ""

                # For SELECT queries, fetch results
                if query_type in ("SELECT", "PRAGMA", "WITH"):
                    rows = cursor.fetchall()
                    # Convert sqlite3.Row objects to dicts
                    return [dict(row) for row in rows]
                else:
                    # For INSERT/UPDATE/DELETE operations
                    # Note: If database is opened in read-only mode, SQLite will
                    # automatically raise an OperationalError for write operations
                    affected_rows = cursor.rowcount
                    if self.connection is not None:
                        self.connection.commit()

                    # Log the operation
                    logger.info(f"Executed {query_type} query, affected rows: {affected_rows}")
                    return []
        except sqlite3.Error as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query: {query[:100]}...")  # Log first 100 chars of query
            raise

    async def aget_relationships(self) -> list[dict[str, Any]]:
        """Get foreign key relationships between tables.

        Returns
        -------
            List of relationship dictionaries
        """
        tables = self._get_all_tables()
        relationships: list[dict[str, Any]] = []

        with self._get_cursor() as cursor:
            for table in tables:
                if not self._validate_identifier(table):
                    continue

                cursor.execute(f'PRAGMA foreign_key_list("{table}")')  # nosec B608 - validated
                fk_rows = cursor.fetchall()

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

        with self._get_cursor() as cursor:
            # Get all indexes
            cursor.execute("""
                SELECT name, tbl_name, sql FROM sqlite_master
                WHERE type='index' AND sql IS NOT NULL
            """)
            index_rows = cursor.fetchall()

            for row in index_rows:
                index_name = row[0]
                table_name = row[1]

                if not self._validate_identifier(index_name, "index"):
                    continue
                if not self._validate_identifier(table_name, "table"):
                    continue

                # Get index info
                cursor.execute(f'PRAGMA index_info("{index_name}")')  # nosec B608 - validated
                col_rows = cursor.fetchall()
                columns = [col_row[2] for col_row in col_rows]

                # Check if unique
                cursor.execute(f'PRAGMA index_list("{table_name}")')  # nosec B608 - validated
                idx_list = cursor.fetchall()
                is_unique = False
                for idx in idx_list:
                    if idx[1] == index_name:
                        is_unique = bool(idx[2])
                        break

                indexes.append(
                    {
                        "index_name": index_name,
                        "table_name": table_name,
                        "columns": columns,
                        "index_type": "btree",  # SQLite primarily uses B-tree
                        "is_unique": is_unique,
                    }
                )

        return indexes

    async def aget_table_statistics(self) -> dict[str, dict[str, Any]]:
        """Get table statistics for query optimization.

        Returns
        -------
            Dictionary mapping table names to statistics
        """
        tables = self._get_all_tables()
        stats = {}

        with self._get_cursor() as cursor:
            for table in tables:
                if not self._validate_identifier(table):
                    continue  # Skip invalid table names

                # Get row count - use quote for identifier
                # SQLite uses double quotes for identifiers
                cursor.execute(f'SELECT COUNT(*) FROM "{table}"')  # nosec B608 - validated
                row_count = cursor.fetchone()[0]

                # Get table size (approximate)
                cursor.execute(
                    "SELECT SUM(LENGTH(sql)) FROM sqlite_master WHERE tbl_name = ?", (table,)
                )
                size_result = cursor.fetchone()[0]
                size_bytes = size_result if size_result else 0

                stats[table] = {
                    "row_count": row_count,
                    "size_bytes": size_bytes,
                    "last_updated": None,  # SQLite doesn't track this
                }

        return stats

    def close(self) -> None:
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.close()

    def __repr__(self) -> str:
        """Return string representation."""
        mode = "read-only" if self.read_only else "read-write"
        return f"SQLiteAdapter(db_path='{self.db_path}', mode='{mode}')"
