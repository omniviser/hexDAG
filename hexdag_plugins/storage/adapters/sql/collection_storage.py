"""SQLAlchemy-based SupportsCollectionStorage with JSON-column tables.

Each collection maps to a database table with schema
``(key TEXT PRIMARY KEY, data JSON NOT NULL)``.  Documents are stored as
JSON blobs; filtering is done in Python after loading all docs.

Usage::

    from hexdag_plugins.storage.adapters.sql.collection_storage import SQLCollectionStorage

    storage = SQLCollectionStorage(connection_string="sqlite+aiosqlite:///hexdag.db")
    await storage.asetup()
    await storage.asave("pipeline_runs", "run-1", {"status": "running"})
    doc = await storage.aload("pipeline_runs", "run-1")
    await storage.aclose()
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine


class SQLCollectionStorage:
    """SQLAlchemy ``SupportsCollectionStorage`` with JSON-column tables.

    Parameters
    ----------
    connection_string : str | None
        Database connection string (e.g. ``"sqlite+aiosqlite:///hexdag.db"``
        or ``"postgresql+asyncpg://user:pass@host/db"``).
    pool_size : int
        Number of permanent connections in the pool (default: 5).
    table_prefix : str
        Prefix for generated table names (default: ``"hexdag_"``).
    """

    def __init__(
        self,
        connection_string: str | None = None,
        pool_size: int = 5,
        table_prefix: str = "hexdag_",
    ) -> None:
        self._connection_string = connection_string
        self._pool_size = pool_size
        self._table_prefix = table_prefix
        self._engine: AsyncEngine | None = None
        self._ensured_tables: set[str] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def asetup(self) -> None:
        """Create the async engine.  Must be called before any CRUD ops."""
        if self._connection_string is None:
            msg = "connection_string is required for SQLCollectionStorage"
            raise ValueError(msg)
        self._engine = create_async_engine(
            self._connection_string,
            pool_size=self._pool_size,
            echo=False,
        )

    async def aclose(self) -> None:
        """Dispose the connection pool."""
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None

    # ------------------------------------------------------------------
    # SupportsCollectionStorage implementation
    # ------------------------------------------------------------------

    async def asave(self, collection: str, key: str, data: dict[str, Any]) -> None:
        """Save a document (upsert semantics)."""
        table = await self._ensure_table(collection)
        data_json = json.dumps(data, default=str)
        async with AsyncSession(self._engine) as session:  # type: ignore[arg-type]
            # SQLite uses INSERT OR REPLACE; PostgreSQL uses ON CONFLICT
            await session.execute(
                text(f"""
                    INSERT INTO {table} (key, data) VALUES (:key, :data)
                    ON CONFLICT (key) DO UPDATE SET data = :data
                """),
                {"key": key, "data": data_json},
            )
            await session.commit()

    async def aload(self, collection: str, key: str) -> dict[str, Any] | None:
        """Load a document by key."""
        table = await self._ensure_table(collection)
        async with AsyncSession(self._engine) as session:  # type: ignore[arg-type]
            result = await session.execute(
                text(f"SELECT data FROM {table} WHERE key = :key"),
                {"key": key},
            )
            row = result.fetchone()
            if row is None:
                return None
            return json.loads(row[0])  # type: ignore[arg-type]

    async def aquery(
        self, collection: str, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Query all documents, optionally filtered in Python."""
        table = await self._ensure_table(collection)
        async with AsyncSession(self._engine) as session:  # type: ignore[arg-type]
            result = await session.execute(text(f"SELECT data FROM {table}"))
            rows = result.fetchall()

        docs = [json.loads(row[0]) for row in rows]
        if filters:
            docs = [d for d in docs if all(d.get(k) == v for k, v in filters.items())]
        return docs

    async def adelete(self, collection: str, key: str) -> bool:
        """Delete a document.  Returns True if it existed."""
        table = await self._ensure_table(collection)
        async with AsyncSession(self._engine) as session:  # type: ignore[arg-type]
            result = await session.execute(
                text(f"DELETE FROM {table} WHERE key = :key"),
                {"key": key},
            )
            await session.commit()
            return result.rowcount > 0  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ensure_table(self, collection: str) -> str:
        """Lazily create the table for a collection if it doesn't exist."""
        table_name = f"{self._table_prefix}{collection}"
        if table_name in self._ensured_tables:
            return table_name

        if self._engine is None:
            msg = "SQLCollectionStorage not set up. Call asetup() first."
            raise RuntimeError(msg)

        async with self._engine.begin() as conn:
            await conn.execute(
                text(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        key TEXT PRIMARY KEY,
                        data TEXT NOT NULL
                    )
                """)
            )
        self._ensured_tables.add(table_name)
        return table_name
