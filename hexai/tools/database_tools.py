"""Database-related tools that require database port."""

import asyncio
import re
from typing import Any

from hexai.core.ports.database import DatabasePort
from hexai.core.ports.llm import LLM, Message
from hexai.core.registry import tool


@tool(
    name=["db_query", "database_query", "sql_query"],
    namespace="core",
    required_ports=["database"],
    description="Execute SQL query on database",
)
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
    """
    if database_port is None:
        raise ValueError("Database port is required but not provided")

    # Use the database port to execute the query
    return await database_port.aexecute_query(sql, params or {})


@tool(
    name=["db_execute", "database_execute", "sql_execute"],
    namespace="core",
    required_ports=["database"],
    description="Execute SQL command on database",
)
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
    """
    if database_port is None:
        raise ValueError("Database port is required but not provided")

    # Execute returns no value, just runs the query
    await database_port.aexecute_query(sql, params or {})


@tool(
    name=["db_search", "database_search", "semantic_search"],
    namespace="core",
    required_ports=["database", "llm"],
    description="Semantic search in database using LLM embeddings",
)
async def database_search(
    query: str,
    table: str,
    column: str,
    limit: int = 10,
    database_port: DatabasePort | None = None,
    llm_port: LLM | None = None,
) -> list[dict[str, Any]]:
    """Perform semantic search in database using LLM for embeddings.

    This advanced tool requires both database and LLM ports.

    Args
    ----
        query: Search query text
        table: Table to search in
        column: Column to search
        limit: Maximum number of results
        database_port: Injected database port
        llm_port: Injected LLM port for embeddings

    Returns
    -------
        List of search results ranked by similarity
    """
    if database_port is None:
        raise ValueError("Database port is required but not provided")
    if llm_port is None:
        raise ValueError("LLM port is required but not provided")

    # Get embedding for query using LLM
    embedding_prompt = [
        Message(role="user", content=f"Generate embedding for search query: {query}")
    ]
    query_embedding = await llm_port.aresponse(embedding_prompt)

    # Validate table and column names to prevent SQL injection
    # Only allow alphanumeric characters and underscores
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table):
        raise ValueError(f"Invalid table name: {table}")
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", column):
        raise ValueError(f"Invalid column name: {column}")

    # Build semantic search SQL (simplified example)
    # In real implementation, this would use vector similarity
    sql = f"""
    SELECT *, similarity({column}, :embedding) as score
    FROM {table}
    ORDER BY score DESC
    LIMIT :limit
    """  # nosec B608 - Table and column names are validated above

    results = await database_port.aexecute_query(
        sql, {"embedding": query_embedding, "limit": limit}
    )

    return results


@tool(
    name=["db_tables", "list_tables", "show_tables"],
    namespace="core",
    required_ports=["database"],
    description="List all tables in database",
)
async def list_tables(database_port: DatabasePort | None = None) -> list[str]:
    """List all tables in the database.

    Args
    ----
        database_port: Injected database port

    Returns
    -------
        List of table names
    """
    if database_port is None:
        raise ValueError("Database port is required but not provided")

    # Standard SQL to get table list (may vary by database)
    sql = """
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
    """

    results = await database_port.aexecute_query(sql)
    return [row["table_name"] for row in results]


@tool(
    name=["db_schema", "table_schema", "describe_table"],
    namespace="core",
    required_ports=["database"],
    description="Get schema information for a table",
)
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
    """
    if database_port is None:
        raise ValueError("Database port is required but not provided")

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
@tool(
    name=["db_query_sync", "database_query_sync"],
    namespace="core",
    required_ports=["database"],
    description="Execute SQL query on database (synchronous)",
)
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
    """
    if database_port is None:
        raise ValueError("Database port is required but not provided")
    result = asyncio.run(database_port.aexecute_query(sql, params or {}))
    return result


# Export key tools
__all__ = [
    "database_query",
    "database_execute",
    "database_search",
    "list_tables",
    "describe_table",
    "database_query_sync",
]
