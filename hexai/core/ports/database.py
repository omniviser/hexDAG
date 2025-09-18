"""Database port interface for accessing database schema information."""

from typing import Any, Protocol


class DatabasePort(Protocol):
    """Port interface for accessing database schema and metadata.

    This port abstracts access to database systems, allowing the analytics engine to work with
    different database backends.
    """

    def get_table_schemas(self) -> dict[str, dict[str, Any]]:
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

    def get_relationships(self) -> list[dict[str, Any]]:
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

    def get_indexes(self) -> list[dict[str, Any]]:
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

    def get_table_statistics(self) -> dict[str, dict[str, Any]]:
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
