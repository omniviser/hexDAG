"""Database port interface for accessing database schema information.

.. deprecated::
    Prefer :class:`~hexdag.kernel.ports.data_store.SupportsQuery` and
    :class:`~hexdag.kernel.ports.data_store.SupportsSchema` for new code.
    ``DatabasePort`` is retained for backward compatibility.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

# Backward-compat re-export: canonical location is hexdag.kernel.ports.vector_search
from hexdag.kernel.ports.vector_search import SupportsVectorSearch

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class ColumnType(Enum):
    """Database column types."""

    TEXT = "text"
    INT = "int"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    TIMESTAMP = "timestamp"
    DECIMAL = "decimal"
    BLOB = "blob"


@dataclass
class ColumnSchema:
    """Schema information for a database column."""

    name: str
    type: ColumnType | str  # ColumnType enum or database-specific type string
    nullable: bool = True
    primary_key: bool = False
    foreign_key: str | None = None  # Format: "table.column"


@dataclass
class TableSchema:
    """Schema information for a database table."""

    name: str
    columns: list[ColumnSchema]


@runtime_checkable
class Database(Protocol):
    """Port interface for accessing database schema and metadata.

    This port abstracts access to database systems, allowing the analytics engine to work with
    different database backends. Implementations may use direct connections (psycopg2, SQLAlchemy)
    or REST APIs for cloud databases (Snowflake, BigQuery, etc.).
    """

    # Required methods
    @abstractmethod
    async def aget_table_schemas(self) -> dict[str, dict[str, Any]]:
        """Get schema information for all tables.

        Returns
        -------
        Dictionary mapping table names to schema information with structure::

            {
                "table_name": {
                    "table_name": str,
                    "columns": dict[str, str],  # column_name -> data_type
                    "primary_keys": list[str],
                    "foreign_keys": list[dict[str, str]]
                }
            }

        """
        ...

    @abstractmethod
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
        ...


@runtime_checkable
class SupportsStreamingQuery(Protocol):
    """Optional protocol for adapters that support streaming queries.

    This protocol enables efficient handling of large result sets by streaming
    rows one at a time rather than loading all results into memory.
    """

    def query(
        self,
        table: str,
        filters: dict[str, Any] | None = None,
        columns: list[str] | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Query rows from a table with optional filtering and column selection.

        Args
        ----
            table: Name of the table to query
            filters: Optional column-value pairs to filter by
            columns: Optional list of columns to return (None = all)
            limit: Optional maximum number of rows to return

        Returns
        -------
            AsyncIterator[dict]: Stream of rows as dictionaries

        Raises
        ------
        ValueError
            If table doesn't exist or filters/columns are invalid
        """
        ...


@runtime_checkable
class SupportsRawSQL(Protocol):
    """Optional protocol for adapters that support raw SQL queries with streaming."""

    async def query_raw(
        self, sql: str, params: dict[str, Any] | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Execute a raw SQL query with streaming results.

        Args
        ----
            sql: SQL query string
            params: Optional query parameters

        Returns
        -------
            AsyncIterator[dict]: Stream of result rows

        Raises
        ------
        ValueError
            If SQL is invalid
        """
        ...


@runtime_checkable
class SupportsReadOnly(Protocol):
    """Optional protocol for adapters that are read-only.

    This protocol marker indicates that a database adapter only supports
    read operations (SELECT queries) and will raise errors on write attempts.
    Useful for analytics databases, reporting systems, or production replicas.

    Adapters implementing this protocol should raise appropriate errors when
    write operations are attempted through aexecute_query().

    Examples
    --------
    Check if adapter is read-only::

        if isinstance(db_adapter, SupportsReadOnly):
            print("This adapter is read-only")
            # Only use SELECT queries
    """

    @abstractmethod
    async def is_read_only(self) -> bool:
        """Check if the adapter is in read-only mode.

        Returns
        -------
            True if adapter is read-only, False otherwise
        """
        ...


# Backward-compat alias (deprecated: use Database)
DatabasePort = Database

__all__ = [
    "ColumnSchema",
    "ColumnType",
    "Database",
    "DatabasePort",
    "SupportsRawSQL",
    "SupportsReadOnly",
    "SupportsStreamingQuery",
    "SupportsVectorSearch",
    "TableSchema",
]
