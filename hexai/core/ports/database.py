"""Database port interface for accessing database schema information."""

from abc import abstractmethod
from typing import Any, Protocol, runtime_checkable

from hexai.core.registry.decorators import port


@port(name="database", namespace="core")
@runtime_checkable
class DatabasePort(Protocol):
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
            Dictionary mapping table names to schema information:
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
            List of relationship dictionaries with structure:
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
            List of index dictionaries with structure:
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
            Dictionary mapping table names to statistics:
            {
                "table_name": {
                    "row_count": int,
                    "size_bytes": int,
                    "last_updated": str
                }
            }
        """
        ...
