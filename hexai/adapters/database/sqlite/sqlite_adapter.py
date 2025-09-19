"""SQLite database adapter implementation."""

import re
import sqlite3
from pathlib import Path
from sqlite3 import Connection
from typing import Any

from hexai.core.registry.decorators import adapter


@adapter(
    name="sqlite",
    implements_port="database",
    description="SQLite database adapter for SQL query execution and schema introspection",
)
class SQLiteAdapter:
    """SQLite adapter for database port.

    Provides a lightweight, file-based database solution that implements
    the DatabasePort interface for SQL execution and schema introspection.
    """

    def __init__(self, db_path: str = "hexdag.db", **kwargs: Any) -> None:
        """Initialize SQLite adapter.

        Args:
            db_path: Path to the SQLite database file
            **kwargs: Additional configuration options
        """
        self.db_path = Path(db_path)
        self.connection: Connection | None = None
        self._ensure_database()

    def _ensure_database(self) -> None:
        """Ensure database connection exists."""
        self.connection = sqlite3.connect(str(self.db_path))
        self.connection.row_factory = sqlite3.Row  # Return rows as dictionaries

    async def aget_table_schemas(self) -> dict[str, dict[str, Any]]:
        """Get schema information for all tables.

        Returns
        -------
            Dictionary mapping table names to schema information
        """
        if self.connection is None:
            raise RuntimeError("Database connection not established")

        cursor = self.connection.cursor()

        # Get all tables
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """)
        tables = [row[0] for row in cursor.fetchall()]

        schemas = {}
        for table in tables:
            # Get column info
            cursor.execute(f'PRAGMA table_info("{table}")')  # nosec B608 - table from sqlite_master
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
            cursor.execute(f'PRAGMA foreign_key_list("{table}")')  # nosec B608 - table from sqlite_master
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
            List of dictionaries representing query result rows
        """
        if self.connection is None:
            raise RuntimeError("Database connection not established")

        cursor = self.connection.cursor()

        # Convert dict params to list for sqlite3
        if params:
            # SQLite uses ? placeholders, need to convert from :name format
            import re

            param_names = re.findall(r":(\w+)", query)
            query = re.sub(r":(\w+)", "?", query)
            param_values = [params.get(name) for name in param_names]
            cursor.execute(query, param_values)
        else:
            cursor.execute(query)

        # For SELECT queries, fetch results
        if query.strip().upper().startswith(("SELECT", "PRAGMA", "WITH")):
            rows = cursor.fetchall()
            # Convert sqlite3.Row objects to dicts
            return [dict(row) for row in rows]
        else:
            # For INSERT/UPDATE/DELETE, commit and return empty list
            self.connection.commit()
            return []

    async def aget_relationships(self) -> list[dict[str, Any]]:
        """Get foreign key relationships between tables.

        Returns
        -------
            List of relationship dictionaries
        """
        if self.connection is None:
            raise RuntimeError("Database connection not established")

        cursor = self.connection.cursor()

        # Get all tables
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """)
        tables = [row[0] for row in cursor.fetchall()]

        relationships: list[dict[str, Any]] = []
        for table in tables:
            cursor.execute(f'PRAGMA foreign_key_list("{table}")')  # nosec B608 - table from sqlite_master
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
        if self.connection is None:
            raise RuntimeError("Database connection not established")

        cursor = self.connection.cursor()

        # Get all indexes
        cursor.execute("""
            SELECT name, tbl_name, sql FROM sqlite_master
            WHERE type='index' AND sql IS NOT NULL
        """)
        index_rows = cursor.fetchall()

        indexes = []
        for row in index_rows:
            index_name = row[0]
            table_name = row[1]

            # Get index info
            cursor.execute(f'PRAGMA index_info("{index_name}")')  # nosec B608 - index from sqlite_master
            col_rows = cursor.fetchall()
            columns = [col_row[2] for col_row in col_rows]

            # Check if unique
            cursor.execute(f'PRAGMA index_list("{table_name}")')  # nosec B608 - table from sqlite_master
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
        if self.connection is None:
            raise RuntimeError("Database connection not established")

        cursor = self.connection.cursor()

        # Get all tables
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """)
        tables = [row[0] for row in cursor.fetchall()]

        stats = {}
        for table in tables:
            # Validate table name to prevent injection (even though it comes from sqlite_master)

            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table):
                continue  # Skip invalid table names

            # Get row count - use quote for identifier
            # SQLite uses double quotes for identifiers
            cursor.execute(f'SELECT COUNT(*) FROM "{table}"')  # nosec B608 - table validated
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
        """String representation."""
        return f"SQLiteAdapter(db_path='{self.db_path}')"
