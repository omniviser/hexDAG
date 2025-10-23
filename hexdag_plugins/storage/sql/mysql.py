"""MySQL adapter using SQLAlchemy with aiomysql driver."""

from hexdag.core.registry.decorators import adapter

from .base import DatabaseConfig, SQLAdapter


@adapter("database", name="mysql", namespace="storage")
class MySQLAdapter(SQLAdapter):
    """MySQL database adapter with async connection pooling.

    Uses SQLAlchemy with aiomysql driver for async MySQL operations.
    Provides connection pooling, health checks, and standard SQL operations.

    Example connection string:
        mysql+aiomysql://user:password@host:port/database

    Configuration:
        - connection_string: MySQL connection URL (env: DATABASE_URL)
        - pool_size: Number of permanent connections (default: 5)
        - max_overflow: Extra connections beyond pool_size (default: 10)
        - pool_timeout: Timeout for getting connection (default: 30.0)
        - pool_recycle: Recycle connections after seconds (default: 3600)
        - pool_pre_ping: Test connections before use (default: True)
    """

    Config = DatabaseConfig
