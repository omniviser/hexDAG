"""Database port interface for accessing database schema information."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from hexdag.core.registry.decorators import port

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from hexdag.core.ports.healthcheck import HealthStatus


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


@port(name="database", namespace="core")
@runtime_checkable
class DatabasePort(Protocol):
    """Port interface for accessing database schema and metadata.

    This port abstracts access to database systems, allowing the analytics engine to work with
    different database backends. Implementations may use direct connections (psycopg2, SQLAlchemy)
    or REST APIs for cloud databases (Snowflake, BigQuery, etc.).

    Optional Methods
    ----------------
    Adapters may optionally implement:
    - ahealth_check(): Verify database connectivity and query execution
    - query(): Streaming query interface with filters
    - aget_relationships(): Foreign key relationships
    - aget_indexes(): Index information
    - aget_table_statistics(): Table statistics
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

    # Optional methods for enhanced functionality
    async def aget_relationships(self) -> list[dict[str, Any]]:
        """Get foreign key relationships between tables.

        Returns
        -------
        List of relationship dictionaries with structure::

            {
                "from_table": str,
                "from_column": str,
                "to_table": str,
                "to_column": str,
                "relationship_type": str  # "many_to_one", etc.
            }

        """
        ...

    async def aget_indexes(self) -> list[dict[str, Any]]:
        """Get index information for performance optimization.

        Returns
        -------
        List of index dictionaries with structure::

            {
                "index_name": str,
                "table_name": str,
                "columns": list[str],
                "index_type": str,  # "btree", "hash", etc.
                "is_unique": bool
            }

        """
        ...

    async def aget_table_statistics(self) -> dict[str, dict[str, Any]]:
        """Get table statistics for query optimization.

        Returns
        -------
        Dictionary mapping table names to statistics with structure::

            {
                "table_name": {
                    "row_count": int,
                    "size_bytes": int,
                    "last_updated": str
                }
            }

        """
        ...

    async def ahealth_check(self) -> HealthStatus:
        """Check database adapter health and connectivity (optional).

        Adapters should verify:
        - Database connection status
        - Connection pool health
        - Basic query execution (e.g., SELECT 1)
        - Authentication status

        This method is optional. If not implemented, the adapter will be
        considered healthy by default.

        Returns
        -------
        HealthStatus
            Current health status with details about database connectivity

        Examples
        --------
        Example usage::

            # PostgreSQL adapter health check
            status = await postgres_adapter.ahealth_check()
            status.status  # "healthy", "degraded", or "unhealthy"
            status.latency_ms  # Time taken for health check query
            status.details  # {"pool_size": 10, "active_connections": 3}
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
class SupportsTableSchema(Protocol):
    """Optional protocol for adapters that support table schema information."""

    async def get_table_schema(self, table: str) -> TableSchema:
        """Get schema information for a table.

        Args
        ----
            table: Name of the table to get schema for
        """
        ...


@runtime_checkable
class SupportsIndexes(Protocol):
    """Optional protocol for adapters that support index information."""

    async def get_indexes(self, table: str) -> list[dict[str, Any]]:
        """Get index information for a table.

        Args
        ----
            table: Name of the table to get index information for
        """
        ...


@runtime_checkable
class SupportsStatistics(Protocol):
    """Optional protocol for adapters that support table statistics."""

    async def get_table_statistics(self, table: str) -> dict[str, Any]:
        """Get table statistics for a table.

        Args
        ----
            table: Name of the table to get statistics for
        """
        ...
