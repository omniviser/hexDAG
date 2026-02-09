"""PostgreSQL adapter using SQLAlchemy with asyncpg driver."""

from .base import SQLAdapter


class PostgreSQLAdapter(SQLAdapter):
    """PostgreSQL database adapter with async connection pooling.

    Uses SQLAlchemy with asyncpg driver for async PostgreSQL operations.
    Provides connection pooling, health checks, and standard SQL operations.

    Parameters
    ----------
    connection_string : str | None
        PostgreSQL connection URL (e.g., postgresql+asyncpg://user:password@host:port/database)
        Auto-resolved from DATABASE_URL env var if not provided
    pool_size : int
        Number of permanent connections (default: 5)
    max_overflow : int
        Extra connections beyond pool_size (default: 10)
    pool_timeout : float
        Timeout for getting connection (default: 30.0)
    pool_recycle : int
        Recycle connections after seconds (default: 3600)
    pool_pre_ping : bool
        Test connections before use (default: True)

    Examples
    --------
    Basic usage::

        adapter = PostgreSQLAdapter(
            connection_string="postgresql+asyncpg://user:pass@localhost/mydb"
        )
        await adapter.asetup()

        # Execute query
        results = await adapter.aexecute_query("SELECT * FROM users")

        # Get table schemas
        schemas = await adapter.aget_table_schemas()

        # Close connection
        await adapter.aclose()
    """

    _hexdag_icon = "Database"
    _hexdag_color = "#336791"  # PostgreSQL blue

    pass
