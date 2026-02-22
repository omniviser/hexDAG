"""Database tools lib — agent-callable database operations.

Each public async ``a*`` method is auto-exposed as an agent tool.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hexdag.kernel.lib_base import HexDAGLib

if TYPE_CHECKING:
    from hexdag.kernel.ports.data_store import SupportsQuery


class DatabaseTools(HexDAGLib):
    """Database query tools for agent nodes.

    Provides SQL query execution, table listing, and schema introspection
    as agent-callable tools.

    Parameters
    ----------
    store : SupportsQuery
        Any adapter implementing the ``SupportsQuery`` protocol
        (e.g. SQLiteAdapter, PgVectorAdapter, SQLAlchemyAdapter).
    """

    def __init__(self, store: SupportsQuery) -> None:
        self._store = store

    async def adatabase_query(
        self, sql: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a SQL query and return results.

        Args
        ----
            sql: SQL query to execute
            params: Optional parameters for parameterised queries

        Returns
        -------
            List of dictionaries representing query result rows
        """
        return await self._store.aexecute_query(sql, params)

    async def alist_tables(self) -> list[str]:
        """List all tables in the database.

        Returns
        -------
            List of table names
        """
        sql = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        """
        results = await self._store.aexecute_query(sql)
        return [row["table_name"] for row in results]

    async def adescribe_table(self, table: str) -> list[dict[str, Any]]:
        """Get column information for a database table.

        Args
        ----
            table: Name of the table to describe

        Returns
        -------
            List of column information dicts
        """
        sql = """
        SELECT
            column_name,
            data_type,
            is_nullable,
            column_default
        FROM information_schema.columns
        WHERE table_name = :table
        ORDER BY ordinal_position
        """
        return await self._store.aexecute_query(sql, {"table": table})
