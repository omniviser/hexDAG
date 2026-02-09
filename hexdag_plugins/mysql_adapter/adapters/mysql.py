"""MySQL database adapter implementation.

This is a document-store style MySQL adapter that provides:
- JSON document storage
- Collection-based organization
- Query filtering via JSON path

For standard SQL operations (raw queries, table schemas), use
hexdag_plugins.storage.adapters.sql.MySQLAdapter instead.
"""

import json
import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Literal

import pymysql
import pymysql.cursors
from pymysql.connections import Connection

logger = logging.getLogger(__name__)

# Convention: MySQL charset options for dropdown menus in Studio UI
MySQLCharset = Literal["utf8mb4", "utf8", "latin1", "ascii"]


class MySQLAdapter:
    """MySQL document-store adapter for hexDAG.

    Provides a document-store interface over MySQL with JSON storage.
    Useful for applications that need document-style access with MySQL backend.

    Parameters
    ----------
    host : str
        MySQL server host (default: "localhost")
    port : int
        MySQL server port (default: 3306)
    user : str
        Database user (default: "root")
    password : str
        Database password (auto-resolved from MYSQL_PASSWORD env var)
    database : str
        Database name to use (default: "hexdag")
    charset : str
        Character set for connection (default: "utf8mb4")

    Examples
    --------
    Basic usage::

        adapter = MySQLAdapter(
            host="localhost",
            user="root",
            password="secret",
            database="myapp"
        )

        # Insert document
        doc_id = await adapter.ainsert("users", {"name": "Alice", "age": 30})

        # Query documents
        users = await adapter.aquery("users", {"age": 30})

        # Get by ID
        user = await adapter.aget("users", doc_id)
    """

    _hexdag_icon = "Database"
    _hexdag_color = "#00758f"  # MySQL teal

    def __init__(
        self,
        host: str = "localhost",
        port: int = 3306,
        user: str = "root",
        password: str | None = None,
        database: str = "hexdag",
        charset: MySQLCharset = "utf8mb4",
        **kwargs: Any,
    ) -> None:
        """Initialize MySQL adapter.

        Parameters
        ----------
        host : str
            MySQL server host (default: "localhost")
        port : int
            MySQL server port (default: 3306)
        user : str
            Database user (default: "root")
        password : str | None
            Database password (auto-resolved from MYSQL_PASSWORD env var)
        database : str
            Database name to use (default: "hexdag")
        charset : str
            Character set for connection (default: "utf8mb4")
        **kwargs : Any
            Additional connection options
        """
        # Resolve password from env if not provided
        resolved_password = password or os.getenv("MYSQL_PASSWORD", "")

        self.connection_params: dict[str, Any] = {
            "host": host,
            "port": port,
            "user": user,
            "password": resolved_password,
            "database": database,
            "charset": charset,
            "cursorclass": pymysql.cursors.DictCursor,
            **kwargs,
        }
        self._ensure_database()
        self._ensure_tables()

    @contextmanager
    def _get_connection(self) -> Generator[Connection, None, None]:
        """Context manager for database connections."""
        connection = pymysql.connect(**self.connection_params)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _ensure_database(self) -> None:
        """Ensure the database exists."""
        # Connect without specifying database
        params = self.connection_params.copy()
        database = params.pop("database")

        try:
            connection = pymysql.connect(**params)
            with connection.cursor() as cursor:
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database}")
            connection.commit()
            connection.close()
        except pymysql.err.OperationalError as e:
            logger.warning(f"Could not create database: {e}")
            # Database might already exist or we don't have permissions

    def _ensure_tables(self) -> None:
        """Ensure required tables exist."""
        with self._get_connection() as connection, connection.cursor() as cursor:
            # Create main document store table
            cursor.execute("""
                    CREATE TABLE IF NOT EXISTS hexdag_documents (
                        id VARCHAR(255) PRIMARY KEY,
                        collection VARCHAR(255) NOT NULL,
                        document JSON NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        INDEX idx_collection (collection),
                        INDEX idx_collection_id (collection, id)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """)

            # Create metadata table
            cursor.execute("""
                    CREATE TABLE IF NOT EXISTS hexdag_metadata (
                        collection VARCHAR(255) PRIMARY KEY,
                        count INT DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """)

    async def ainsert(self, collection: str, data: dict[str, Any]) -> str:
        """Insert data into collection.

        Parameters
        ----------
        collection : str
            Collection/table name
        data : dict[str, Any]
            Data to insert

        Returns
        -------
        str
            ID of inserted record
        """
        import asyncio

        loop = asyncio.get_event_loop()

        def insert() -> str:
            with self._get_connection() as connection, connection.cursor() as cursor:
                # Generate ID if not provided
                doc_id = data.get("_id") or data.get("id") or self._generate_id()
                data["_id"] = doc_id

                # Insert document
                cursor.execute(
                    """
                        INSERT INTO hexdag_documents (id, collection, document)
                        VALUES (%s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                        document = VALUES(document),
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (doc_id, collection, json.dumps(data)),
                )

                # Update collection metadata
                cursor.execute(
                    """
                        INSERT INTO hexdag_metadata (collection, count)
                        VALUES (%s, 1)
                        ON DUPLICATE KEY UPDATE
                        count = count + 1,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (collection,),
                )

                return str(doc_id)

        return await loop.run_in_executor(None, insert)

    async def aget(self, collection: str, id: str) -> dict[str, Any] | None:
        """Get document by ID.

        Parameters
        ----------
        collection : str
            Collection/table name
        id : str
            Document ID

        Returns
        -------
        dict[str, Any] | None
            Document data or None if not found
        """
        import asyncio

        loop = asyncio.get_event_loop()

        def get() -> dict[str, Any] | None:
            with self._get_connection() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """
                        SELECT document FROM hexdag_documents
                        WHERE collection = %s AND id = %s
                    """,
                    (collection, str(id)),
                )
                result = cursor.fetchone()
                if result:
                    doc: dict[str, Any] = json.loads(result["document"])
                    return doc
                return None

        return await loop.run_in_executor(None, get)

    async def aquery(
        self, collection: str, filter: dict[str, Any] | None = None, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Query documents from collection.

        Parameters
        ----------
        collection : str
            Collection/table name
        filter : dict[str, Any] | None
            Query filter (JSON path based)
        limit : int | None
            Maximum number of results

        Returns
        -------
        list[dict[str, Any]]
            List of matching documents
        """
        import asyncio

        loop = asyncio.get_event_loop()

        def query() -> list[dict[str, Any]]:
            with self._get_connection() as connection, connection.cursor() as cursor:
                sql = "SELECT document FROM hexdag_documents WHERE collection = %s"
                params: list[Any] = [collection]

                # Add JSON-based filtering
                if filter:
                    for key, value in filter.items():
                        sql += " AND JSON_EXTRACT(document, %s) = %s"
                        params.extend([f"$.{key}", json.dumps(value)])

                # Add limit
                if limit:
                    sql += f" LIMIT {limit}"

                cursor.execute(sql, params)
                results: list[dict[str, Any]] = []
                for row in cursor.fetchall():
                    doc: dict[str, Any] = json.loads(row["document"])
                    results.append(doc)
                return results

        return await loop.run_in_executor(None, query)

    async def aupdate(self, collection: str, id: str, data: dict[str, Any]) -> bool:
        """Update document in collection.

        Parameters
        ----------
        collection : str
            Collection/table name
        id : str
            Document ID
        data : dict[str, Any]
            Updated data (will be merged with existing)

        Returns
        -------
        bool
            True if updated, False if not found
        """
        import asyncio

        loop = asyncio.get_event_loop()

        def update() -> bool:
            with self._get_connection() as connection, connection.cursor() as cursor:
                # Get existing document
                cursor.execute(
                    """
                        SELECT document FROM hexdag_documents
                        WHERE collection = %s AND id = %s
                    """,
                    (collection, str(id)),
                )
                result = cursor.fetchone()

                if not result:
                    return False

                # Merge with existing data
                existing = json.loads(result["document"])
                existing.update(data)
                existing["_id"] = str(id)

                # Update document
                cursor.execute(
                    """
                        UPDATE hexdag_documents
                        SET document = %s, updated_at = CURRENT_TIMESTAMP
                        WHERE collection = %s AND id = %s
                    """,
                    (json.dumps(existing), collection, str(id)),
                )

                return bool(cursor.rowcount > 0)

        return await loop.run_in_executor(None, update)

    async def adelete(self, collection: str, id: str) -> bool:
        """Delete document from collection.

        Parameters
        ----------
        collection : str
            Collection/table name
        id : str
            Document ID

        Returns
        -------
        bool
            True if deleted, False if not found
        """
        import asyncio

        loop = asyncio.get_event_loop()

        def delete() -> bool:
            with self._get_connection() as connection, connection.cursor() as cursor:
                # Delete document
                cursor.execute(
                    """
                        DELETE FROM hexdag_documents
                        WHERE collection = %s AND id = %s
                    """,
                    (collection, str(id)),
                )

                if cursor.rowcount > 0:
                    # Update collection metadata
                    cursor.execute(
                        """
                            UPDATE hexdag_metadata
                            SET count = GREATEST(0, count - 1),
                                updated_at = CURRENT_TIMESTAMP
                            WHERE collection = %s
                        """,
                        (collection,),
                    )
                    return True
                return False

        return await loop.run_in_executor(None, delete)

    async def alist_collections(self) -> list[str]:
        """List all collections in database.

        Returns
        -------
        list[str]
            List of collection names
        """
        import asyncio

        loop = asyncio.get_event_loop()

        def list_collections() -> list[str]:
            with self._get_connection() as connection, connection.cursor() as cursor:
                cursor.execute("SELECT DISTINCT collection FROM hexdag_documents")
                return [row["collection"] for row in cursor.fetchall()]

        return await loop.run_in_executor(None, list_collections)

    async def acount(self, collection: str) -> int:
        """Count documents in collection.

        Parameters
        ----------
        collection : str
            Collection name

        Returns
        -------
        int
            Number of documents
        """
        import asyncio

        loop = asyncio.get_event_loop()

        def count() -> int:
            with self._get_connection() as connection, connection.cursor() as cursor:
                cursor.execute(
                    "SELECT COUNT(*) as count FROM hexdag_documents WHERE collection = %s",
                    (collection,),
                )
                result = cursor.fetchone()
                return int(result["count"] if result else 0)

        return await loop.run_in_executor(None, count)

    async def adrop_collection(self, collection: str) -> bool:
        """Drop entire collection.

        Parameters
        ----------
        collection : str
            Collection name to drop

        Returns
        -------
        bool
            True if dropped, False if not found
        """
        import asyncio

        loop = asyncio.get_event_loop()

        def drop() -> bool:
            with self._get_connection() as connection, connection.cursor() as cursor:
                # Delete all documents
                cursor.execute("DELETE FROM hexdag_documents WHERE collection = %s", (collection,))

                if cursor.rowcount > 0:
                    # Remove metadata
                    cursor.execute(
                        "DELETE FROM hexdag_metadata WHERE collection = %s",
                        (collection,),
                    )
                return bool(cursor.rowcount > 0)

        return await loop.run_in_executor(None, drop)

    def _generate_id(self) -> str:
        """Generate a unique document ID."""
        import uuid

        return str(uuid.uuid4())

    def __repr__(self) -> str:
        """String representation."""
        host = self.connection_params["host"]
        database = self.connection_params["database"]
        return f"MySQLAdapter(host='{host}', database='{database}')"
