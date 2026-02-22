"""Base SQL adapter using SQLAlchemy for connection pooling and async operations."""

import os
from typing import Any

from hexdag.kernel.ports.database import DatabasePort
from hexdag.kernel.ports.healthcheck import HealthStatus
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine


class SQLAdapter(DatabasePort):
    """Base SQL adapter with SQLAlchemy connection pooling.

    This base class provides common functionality for all SQL databases:
    - Async connection pooling
    - Health checks
    - Basic query execution
    - Proper resource cleanup

    Parameters
    ----------
    connection_string : str | None
        Database connection string (e.g., "postgresql+asyncpg://user:pass@host/db")
        Auto-resolved from DATABASE_URL env var if not provided
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

    def __init__(
        self,
        connection_string: str | None = None,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: float = 30.0,
        pool_recycle: int = 3600,
        pool_pre_ping: bool = True,
    ) -> None:
        """Initialize SQL adapter with configuration.

        Parameters
        ----------
        connection_string : str | None
            Database connection string (auto-resolved from DATABASE_URL env var)
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
        self._connection_string = connection_string or os.getenv("DATABASE_URL")
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._pool_timeout = pool_timeout
        self._pool_recycle = pool_recycle
        self._pool_pre_ping = pool_pre_ping
        self._engine: AsyncEngine | None = None

    async def asetup(self) -> None:
        """Initialize SQLAlchemy async engine with connection pool."""
        if self._connection_string is None:
            msg = "connection_string is required for SQL adapter"
            raise ValueError(msg)

        self._engine = create_async_engine(
            self._connection_string,
            pool_size=self._pool_size,
            max_overflow=self._max_overflow,
            pool_timeout=self._pool_timeout,
            pool_recycle=self._pool_recycle,
            pool_pre_ping=self._pool_pre_ping,
            echo=False,
        )

    async def aclose(self) -> None:
        """Close database connection pool."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None

    async def aget_table_schemas(self) -> dict[str, dict[str, Any]]:
        """Get schema information for all tables.

        Returns
        -------
        dict[str, dict[str, Any]]
            Dictionary mapping table names to schema information
        """
        if not self._engine:
            msg = "Adapter not set up. Call asetup() first."
            raise RuntimeError(msg)

        async with AsyncSession(self._engine) as session:
            # Get all tables
            result = await session.execute(
                text("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                """)
            )
            tables = [row[0] for row in result.fetchall()]

            schemas = {}
            for table in tables:
                # Get columns
                columns_result = await session.execute(
                    text("""
                        SELECT column_name, data_type, is_nullable
                        FROM information_schema.columns
                        WHERE table_name = :table_name
                    """),
                    {"table_name": table},
                )
                columns = {row[0]: row[1] for row in columns_result.fetchall()}

                # Get primary keys
                pk_result = await session.execute(
                    text("""
                        SELECT a.attname
                        FROM pg_index i
                        JOIN pg_attribute a ON a.attrelid = i.indrelid
                            AND a.attnum = ANY(i.indkey)
                        WHERE i.indrelid = :table_name::regclass
                            AND i.indisprimary
                    """),
                    {"table_name": table},
                )
                primary_keys = [row[0] for row in pk_result.fetchall()]

                schemas[table] = {
                    "table_name": table,
                    "columns": columns,
                    "primary_keys": primary_keys,
                    "foreign_keys": [],
                }

            return schemas

    async def aexecute_query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a SQL query and return results.

        Parameters
        ----------
        query : str
            SQL query to execute
        params : dict[str, Any] | None
            Query parameters for parameterized queries

        Returns
        -------
        list[dict[str, Any]]
            List of dictionaries representing query result rows
        """
        if not self._engine:
            msg = "Adapter not set up. Call asetup() first."
            raise RuntimeError(msg)

        async with AsyncSession(self._engine) as session:
            result = await session.execute(text(query), params or {})
            await session.commit()
            rows = result.fetchall()
            return [dict(row._mapping) for row in rows]

    async def aexecute(self, query: str, params: dict[str, Any] | None = None) -> Any:
        """Execute a SQL query (INSERT, UPDATE, DELETE, DDL).

        Parameters
        ----------
        query : str
            SQL query string
        params : dict[str, Any] | None
            Query parameters for parameterized queries

        Returns
        -------
        Any
            Result of the execution
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

        Parameters
        ----------
        query : str
            SQL query string
        params : dict[str, Any] | None
            Query parameters for parameterized queries

        Returns
        -------
        dict[str, Any] | None
            Single row as dictionary or None if no results
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

        Parameters
        ----------
        query : str
            SQL query string
        params : dict[str, Any] | None
            Query parameters for parameterized queries

        Returns
        -------
        list[dict[str, Any]]
            List of rows as dictionaries
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

        Returns
        -------
        HealthStatus
            Health status with database availability and connection pool info
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
