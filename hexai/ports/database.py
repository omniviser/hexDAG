"""Database port protocols for hexDAG adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence


@dataclass
class ColumnSchema:
    """Schema information for a database column."""

    name: str
    type: str
    nullable: bool = True
    primary_key: bool = False
    foreign_key: str | None = None  # Format: "table.column"


@dataclass
class TableSchema:
    """Schema information for a database table."""

    name: str
    columns: list[ColumnSchema]


@runtime_checkable
class DatabasePort(Protocol):
    """Core database functionality that all adapters must implement."""

    async def get_table_schemas(self) -> Sequence[TableSchema]:
        """
        Get schema information for all tables.

        Returns:
            Sequence[TableSchema]: List of table schemas with their columns
        """
        ...

    def query(
        self,
        table: str,
        filters: dict[str, Any] | None = None,
        columns: list[str] | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Query rows from a table with optional filtering and column selection.

        Args:
            table: Name of the table to query
            filters: Optional column-value pairs to filter by
            columns: Optional list of columns to return (None = all)
            limit: Optional maximum number of rows to return

        Returns:
            AsyncIterator[dict]: Stream of rows as dictionaries

        Raises:
            ValueError: If table doesn't exist or filters/columns are invalid
        """
        ...


@runtime_checkable
class SupportsRawSQL(Protocol):
    """Optional protocol for adapters that support raw SQL queries."""

    def query_raw(
        self, sql: str, params: dict[str, Any] | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Execute a raw SQL query.

        Args:
            sql: SQL query string
            params: Optional query parameters

        Returns:
            AsyncIterator[dict]: Stream of result rows

        Raises:
            ValueError: If SQL is invalid
        """
        ...


@runtime_checkable
class SupportsIndexes(Protocol):
    """Optional protocol for adapters that support indexes."""

    async def get_indexes(self, table: str) -> list[str]:
        """
        Get index information for a table.

        Args:
            table: Table name

        Returns:
            list[str]: List of index names

        Raises:
            ValueError: If table doesn't exist
        """
        ...


@runtime_checkable
class SupportsStatistics(Protocol):
    """Optional protocol for adapters that support table statistics."""

    async def get_table_statistics(self, table: str) -> dict[str, int]:
        """
        Get statistical information about a table.

        Args:
            table: Table name

        Returns:
            dict[str, int]: Statistics (e.g., row count)

        Raises:
            ValueError: If table doesn't exist
        """
        ...


@runtime_checkable
class SupportsRelationships(Protocol):
    """Optional protocol for adapters that support foreign key relationships."""

    async def get_relationships(self, table: str) -> list[tuple[str, str, str]]:
        """
        Get foreign key relationships for a table.

        Args:
            table: Table name

        Returns:
            list[tuple[str, str, str]]: List of (source_col, target_table, target_col)

        Raises:
            ValueError: If table doesn't exist
        """
        ...
