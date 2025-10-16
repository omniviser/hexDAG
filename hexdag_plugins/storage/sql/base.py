"""Base SQL adapter using SQLAlchemy for connection pooling and async operations."""

from typing import Any

from hexdag.core import AdapterConfig, ConfigurableAdapter, SecretField
from hexdag.core.ports.healthcheck import HealthStatus
from pydantic import SecretStr
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine


class DatabaseConfig(AdapterConfig):
    """Base configuration for SQL database adapters.

    Attributes
    ----------
    connection_string : SecretStr
        Database connection string (e.g., "postgresql+asyncpg://user:pass@host/db")
    pool_size : int
        Number of permanent connections in the pool (default: 5)
    max_overflow : int
        Maximum number of connections beyond pool_size (default: 10)
    pool_timeout : float
        Timeout in seconds when getting connection from pool (default: 30.0)
    pool_recycle : int
        Recycle connections after this many seconds (default: 3600)
    pool_pre_ping : bool
        Test connections before using them (default: True)
    """

    connection_string: SecretStr = SecretField(
        env_var="DATABASE_URL",
        description="Database connection string",
    )
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: float = 30.0
    pool_recycle: int = 3600
    pool_pre_ping: bool = True


class SQLAdapter(ConfigurableAdapter):
    """Base SQL adapter with SQLAlchemy connection pooling.

    This base class provides common functionality for all SQL databases:
    - Async connection pooling
    - Health checks
    - Basic query execution
    - Proper resource cleanup
    """

    Config = DatabaseConfig

    def __init__(self, **kwargs):
        """Initialize SQL adapter with configuration."""
        super().__init__(**kwargs)
        self._engine: AsyncEngine | None = None

    async def asetup(self):
        """Initialize SQLAlchemy async engine with connection pool."""
        self._engine = create_async_engine(
            self.config.connection_string.get_secret_value(),
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            pool_timeout=self.config.pool_timeout,
            pool_recycle=self.config.pool_recycle,
            pool_pre_ping=self.config.pool_pre_ping,
            echo=False,
        )

    async def aclose(self):
        """Close database connection pool."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None

    async def aexecute(self, query: str, params: dict[str, Any] | None = None) -> Any:
        """Execute a SQL query (INSERT, UPDATE, DELETE, DDL).

        Args:
            query: SQL query string
            params: Query parameters for parameterized queries

        Returns:
            Result of the execution

        Raises:
            RuntimeError: If adapter not set up
        """
        if not self._engine:
            msg = "Adapter not set up. Call asetup() first."
            raise RuntimeError(msg)

        async with AsyncSession(self._engine) as session:
            result = await session.execute(text(query), params or {})
            await session.commit()
            return result

    async def afetch_one(
        self, query: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Fetch a single row from the database.

        Args:
            query: SQL query string
            params: Query parameters for parameterized queries

        Returns:
            Single row as dictionary or None if no results

        Raises:
            RuntimeError: If adapter not set up
        """
        if not self._engine:
            msg = "Adapter not set up. Call asetup() first."
            raise RuntimeError(msg)

        async with AsyncSession(self._engine) as session:
            result = await session.execute(text(query), params or {})
            row = result.fetchone()
            return dict(row._mapping) if row else None

    async def afetch_all(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Fetch all rows from the database.

        Args:
            query: SQL query string
            params: Query parameters for parameterized queries

        Returns:
            List of rows as dictionaries

        Raises:
            RuntimeError: If adapter not set up
        """
        if not self._engine:
            msg = "Adapter not set up. Call asetup() first."
            raise RuntimeError(msg)

        async with AsyncSession(self._engine) as session:
            result = await session.execute(text(query), params or {})
            rows = result.fetchall()
            return [dict(row._mapping) for row in rows]

    async def ahealth_check(self) -> HealthStatus:
        """Check database connection health.

        Returns:
            HealthStatus with database availability and connection pool info
        """
        if not self._engine:
            return HealthStatus(
                status="unhealthy",
                adapter_name=self.__class__.__name__,
                port_name="database",
                details={"message": "Database not initialized"},
            )

        try:
            # Test query
            async with AsyncSession(self._engine) as session:
                await session.execute(text("SELECT 1"))

            # Get pool statistics
            pool = self._engine.pool
            return HealthStatus(
                status="healthy",
                adapter_name=self.__class__.__name__,
                port_name="database",
                details={
                    "pool_size": pool.size(),
                    "checked_in": pool.checkedin(),
                    "checked_out": pool.checkedout(),
                    "overflow": pool.overflow(),
                },
            )
        except Exception as e:
            return HealthStatus(
                status="unhealthy",
                adapter_name=self.__class__.__name__,
                port_name="database",
                error=e,
                details={"error": str(e)},
            )
