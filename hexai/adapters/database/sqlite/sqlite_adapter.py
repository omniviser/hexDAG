"""SQLite database adapter implementation."""

import json
import sqlite3
from pathlib import Path
from sqlite3 import Connection
from typing import Any, cast

from hexai.core.registry.decorators import adapter


@adapter(
    name="sqlite",
    implements_port="database",
    namespace="database",
    description="SQLite database adapter for local file-based storage",
)
class SQLiteAdapter:
    """SQLite adapter for database port.

    Provides a lightweight, file-based database solution perfect for
    development, testing, and small-scale deployments.
    """

    def __init__(self, db_path: str = "hexdag.db", **kwargs: Any) -> None:
        """Initialize SQLite adapter.

        Args:
            db_path: Path to the SQLite database file
            **kwargs: Additional configuration options
        """
        self.db_path = Path(db_path)
        self.connection: Connection | None = None
        self._ensure_database()

    def _ensure_database(self) -> None:
        """Ensure database and tables exist."""
        self.connection = sqlite3.connect(str(self.db_path))
        assert self.connection is not None
        cursor = self.connection.cursor()

        # Create a generic key-value store table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hexdag_store (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collection TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(collection, key)
            )
        """)

        # Create index for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_collection_key
            ON hexdag_store(collection, key)
        """)

        self.connection.commit()

    async def ainsert(self, collection: str, data: dict[str, Any]) -> str:
        """Insert data into collection.

        Args:
            collection: Collection/table name
            data: Data to insert

        Returns:
            ID of inserted record
        """
        assert self.connection is not None
        cursor = self.connection.cursor()

        # Get next ID for this collection
        cursor.execute("SELECT MAX(id) FROM hexdag_store WHERE collection = ?", (collection,))
        max_id = cursor.fetchone()[0]
        next_id = (max_id or 0) + 1
        key = data.get("_id") or data.get("id") or str(next_id)

        # Serialize data to JSON
        value_json = json.dumps(data)

        # Insert or replace
        cursor.execute(
            """
            INSERT OR REPLACE INTO hexdag_store (collection, key, value)
            VALUES (?, ?, ?)
        """,
            (collection, str(key), value_json),
        )

        self.connection.commit()
        return str(key)

    async def aget(self, collection: str, id: str) -> dict[str, Any] | None:
        """Get document by ID.

        Args:
            collection: Collection/table name
            id: Document ID

        Returns:
            Document data or None if not found
        """
        assert self.connection is not None
        cursor = self.connection.cursor()

        cursor.execute(
            """
            SELECT value FROM hexdag_store
            WHERE collection = ? AND key = ?
        """,
            (collection, str(id)),
        )

        result = cursor.fetchone()
        if result:
            data: dict[str, Any] = json.loads(result[0])
            return data
        return None

    async def aquery(
        self, collection: str, filter: dict[str, Any] | None = None, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Query documents from collection.

        Args:
            collection: Collection/table name
            filter: Query filter (basic key-value matching on JSON fields)
            limit: Maximum number of results

        Returns:
            List of matching documents
        """
        assert self.connection is not None
        cursor = self.connection.cursor()

        query = "SELECT value FROM hexdag_store WHERE collection = ?"
        params: tuple[Any, ...] = (collection,)

        # Basic filter support (exact match on JSON fields)
        if filter:
            for key, value in filter.items():
                query += f" AND json_extract(value, '$.{key}') = ?"
                # json_extract returns unquoted strings for string values
                params = params + (value,)

        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query, params)

        results = [json.loads(row[0]) for row in cursor.fetchall()]
        return results

    async def aupdate(self, collection: str, id: str, data: dict[str, Any]) -> bool:
        """Update document in collection.

        Args:
            collection: Collection/table name
            id: Document ID
            data: Updated data

        Returns:
            True if updated, False if not found
        """
        assert self.connection is not None
        cursor = self.connection.cursor()

        # Get existing document
        existing = await self.aget(collection, id)
        if not existing:
            return False

        # Merge data
        existing.update(data)

        # Update in database
        cursor.execute(
            """
            UPDATE hexdag_store
            SET value = ?, updated_at = CURRENT_TIMESTAMP
            WHERE collection = ? AND key = ?
        """,
            (json.dumps(existing), collection, str(id)),
        )

        self.connection.commit()
        return bool(cursor.rowcount > 0)

    async def adelete(self, collection: str, id: str) -> bool:
        """Delete document from collection.

        Args:
            collection: Collection/table name
            id: Document ID

        Returns:
            True if deleted, False if not found
        """
        assert self.connection is not None
        cursor = self.connection.cursor()

        cursor.execute(
            """
            DELETE FROM hexdag_store
            WHERE collection = ? AND key = ?
        """,
            (collection, str(id)),
        )

        self.connection.commit()
        return bool(cursor.rowcount > 0)

    async def alist_collections(self) -> list[str]:
        """List all collections in database.

        Returns:
            List of collection names
        """
        assert self.connection is not None
        cursor = self.connection.cursor()

        cursor.execute("SELECT DISTINCT collection FROM hexdag_store")
        return [row[0] for row in cursor.fetchall()]

    async def acount(self, collection: str) -> int:
        """Count documents in collection.

        Args:
            collection: Collection name

        Returns:
            Number of documents
        """
        assert self.connection is not None
        cursor = self.connection.cursor()

        cursor.execute(
            """
            SELECT COUNT(*) FROM hexdag_store
            WHERE collection = ?
        """,
            (collection,),
        )

        result = cursor.fetchone()
        return cast("int", result[0] if result else 0)

    def close(self) -> None:
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.close()

    def __repr__(self) -> str:  # noqa: D105
        """String representation."""
        return f"SQLiteAdapter(db_path='{self.db_path}')"
