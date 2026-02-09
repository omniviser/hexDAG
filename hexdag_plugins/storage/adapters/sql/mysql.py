"""MySQL adapter using SQLAlchemy with aiomysql driver."""

from .base import SQLAdapter


class MySQLAdapter(SQLAdapter):
    """MySQL database adapter with async connection pooling.

    Uses SQLAlchemy with aiomysql driver for async MySQL operations.
    Provides connection pooling, health checks, and standard SQL operations.

    Parameters
    ----------
    connection_string : str | None
        MySQL connection URL (e.g., mysql+aiomysql://user:password@host:port/database)
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

        adapter = MySQLAdapter(
            connection_string="mysql+aiomysql://user:pass@localhost/mydb"
        )
        await adapter.asetup()

        # Execute query
        results = await adapter.aexecute_query("SELECT * FROM users")

        # Close connection
        await adapter.aclose()
    """

    _hexdag_icon = "Database"
    _hexdag_color = "#00758f"  # MySQL teal

    pass
