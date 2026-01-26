"""Database-related tools that require database port."""

import asyncio
from typing import Any

from hexdag.core.exceptions import DependencyError
from hexdag.core.ports.database import DatabasePort


async def database_query(
    sql: str, params: dict[str, Any] | None = None, database_port: DatabasePort | None = None
) -> list[dict[str, Any]]:
    """Execute a SQL query on the database.

    This tool requires a database port to be available.
    The database port will be injected when the tool is executed.

    Args
    ----
        sql: SQL query to execute
        params: Optional parameters for parameterized queries
        database_port: Injected database port (provided by framework)

    Returns
    -------
        List of dictionaries representing query results

    Raises
    ------
    DependencyError
        If database port is not provided
    """
    if database_port is None:
        raise DependencyError("database_port", "required for database query execution")

    # Use the database port to execute the query
    return await database_port.aexecute_query(sql, params or {})


async def database_execute(
    sql: str, params: dict[str, Any] | None = None, database_port: Any | None = None
) -> None:
    """Execute a SQL command (INSERT, UPDATE, DELETE) on the database.

    This tool requires a database port to be available.

    Args
    ----
        sql: SQL command to execute
        params: Optional parameters for parameterized queries
        database_port: Injected database port (provided by framework)

    Raises
    ------
    DependencyError
        If database port is not provided
    """
    if database_port is None:
        raise DependencyError("database_port", "required for database query execution")

    # Execute returns no value, just runs the query
    await database_port.aexecute_query(sql, params or {})


async def list_tables(database_port: DatabasePort | None = None) -> list[str]:
    """List all tables in the database.

    Args
    ----
        database_port: Injected database port

    Returns
    -------
        List of table names

    Raises
    ------
    DependencyError
        If database port is not provided
    """
    if database_port is None:
        raise DependencyError("database_port", "required for database query execution")

    # Standard SQL to get table list (may vary by database)
    sql = """
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
    """

    results = await database_port.aexecute_query(sql)
    return [row["table_name"] for row in results]


async def describe_table(
    table: str, database_port: DatabasePort | None = None
) -> list[dict[str, Any]]:
    """Get schema information for a database table.

    Args
    ----
        table: Name of the table
        database_port: Injected database port

    Returns
    -------
        List of column information

    Raises
    ------
    DependencyError
        If database port is not provided
    """
    if database_port is None:
        raise DependencyError("database_port", "required for database query execution")

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

    return await database_port.aexecute_query(sql, {"table": table})


# Synchronous versions for compatibility
def database_query_sync(
    sql: str, params: dict[str, Any] | None = None, database_port: DatabasePort | None = None
) -> list[dict[str, Any]]:
    """Execute a database query synchronously.

    Args
    ----
        sql: SQL query to execute
        params: Optional parameters
        database_port: Injected database port

    Returns
    -------
        Query results

    Raises
    ------
    DependencyError
        If database port is not provided
    """
    if database_port is None:
        raise DependencyError("database_port", "required for database query execution")
    return asyncio.run(database_port.aexecute_query(sql, params or {}))


# Export key tools
__all__ = [
    "database_query",
    "database_execute",
    "list_tables",
    "describe_table",
    "database_query_sync",
]
