"""PostgreSQL with pgvector extension adapter for vector similarity search."""

from typing import Any

import asyncpg

from hexdag.core.logging import get_logger

logger = get_logger(__name__)


class PgVectorAdapter:
    """PostgreSQL adapter with pgvector extension support.

    This adapter provides integration with PostgreSQL databases that have the
    pgvector extension installed, enabling vector similarity search alongside
    traditional SQL operations.

    Secret Management
    -----------------
    Password resolution order:
    1. Explicit parameter: PgVectorAdapter(password="...")
    2. Environment variable: PGVECTOR_PASSWORD
    3. Memory port (orchestrator): secret:PGVECTOR_PASSWORD

    Requirements
    ------------
    - PostgreSQL with pgvector extension installed
    - asyncpg Python package
    """

    def __init__(
        self,
        password: str,  # ← Auto-resolved by @adapter decorator
        host: str = "localhost",
        port: int = 5432,
        database: str = "postgres",
        user: str = "postgres",
        read_only: bool = False,
        **kwargs: Any,
    ):
        """Initialize pgvector adapter.

        Parameters
        ----------
        password : str
            PostgreSQL password (auto-resolved from PGVECTOR_PASSWORD env var)
        host : str, default="localhost"
            PostgreSQL host
        port : int, default=5432
            PostgreSQL port
        database : str, default="postgres"
            Database name
        user : str, default="postgres"
            PostgreSQL user
        read_only : bool, default=False
            If True, prevents all write operations (INSERT, UPDATE, DELETE)
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.read_only = read_only
        self._extra_kwargs = kwargs

        self.pool: asyncpg.Pool | None = None

    def _check_read_only(self, operation: str) -> None:
        """Check if adapter is in read-only mode and raise if attempting write.

        Parameters
        ----------
        operation : str
            Name of the operation being attempted (for error message)

        Raises
        ------
        RuntimeError
            If adapter is in read-only mode
        """
        if self.read_only:
            raise RuntimeError(
                f"Cannot perform {operation}: adapter is in read-only mode. "
                f"Set read_only=False to enable write operations."
            )

    async def connect(self) -> None:
        """Establish database connection pool."""
        self.pool = await asyncpg.create_pool(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password,
            min_size=1,
            max_size=10,
        )

        # Verify pgvector extension is available
        async with self.pool.acquire() as conn:
            try:
                if not self.read_only:
                    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    logger.info("pgvector extension is available")
                else:
                    # Just check if extension exists in read-only mode
                    result = await conn.fetchval(
                        "SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector'"
                    )
                    if result:
                        logger.info("pgvector extension is available (read-only mode)")
                    else:
                        logger.warning("pgvector extension not found (read-only mode)")
            except Exception as e:
                logger.warning(f"Could not verify pgvector extension: {e}")

    async def aclose(self) -> None:
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None

    # Backward-compatible alias
    disconnect = aclose

    async def aget_table_schemas(self) -> dict[str, dict[str, Any]]:
        """Get schema information for all tables.

        Returns
        -------
            Dictionary mapping table names to schema information
        """
        if not self.pool:
            raise RuntimeError("Not connected to database")

        async with self.pool.acquire() as conn:
            # Get all tables
            tables = await conn.fetch("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_type = 'BASE TABLE'
            """)

            schemas = {}
            for table_row in tables:
                table_name = table_row["table_name"]

                # Get columns
                columns_query = await conn.fetch(
                    """
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = $1
                    ORDER BY ordinal_position
                    """,
                    table_name,
                )

                columns = {row["column_name"]: row["data_type"] for row in columns_query}

                # Get primary keys
                pk_query = await conn.fetch(
                    """
                    SELECT a.attname
                    FROM pg_index i
                    JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
                    WHERE i.indrelid = $1::regclass AND i.indisprimary
                    """,
                    table_name,
                )
                primary_keys = [row["attname"] for row in pk_query]

                # Get foreign keys
                fk_query = await conn.fetch(
                    """
                    SELECT
                        kcu.column_name,
                        ccu.table_name AS foreign_table_name,
                        ccu.column_name AS foreign_column_name
                    FROM information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                        ON tc.constraint_name = kcu.constraint_name
                        AND tc.table_schema = kcu.table_schema
                    JOIN information_schema.constraint_column_usage AS ccu
                        ON ccu.constraint_name = tc.constraint_name
                        AND ccu.table_schema = tc.table_schema
                    WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = $1
                    """,
                    table_name,
                )

                foreign_keys = [
                    {
                        "from_column": row["column_name"],
                        "to_table": row["foreign_table_name"],
                        "to_column": row["foreign_column_name"],
                    }
                    for row in fk_query
                ]

                schemas[table_name] = {
                    "table_name": table_name,
                    "columns": columns,
                    "primary_keys": primary_keys,
                    "foreign_keys": foreign_keys,
                }

            return schemas

    async def aexecute_query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a SQL query and return results.

        Args
        ----
            query: SQL query to execute
            params: Optional query parameters for safe parameterized queries

        Returns
        -------
            List of dictionaries representing query result rows

        Raises
        ------
        RuntimeError
            If attempting write operation in read-only mode
        """
        # Check for write operations in read-only mode (before connection check)
        query_upper = query.strip().upper()
        write_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE"]
        if self.read_only and any(query_upper.startswith(kw) for kw in write_keywords):
            self._check_read_only(f"query: {query_upper.split()[0]}")

        if not self.pool:
            raise RuntimeError("Not connected to database")

        async with self.pool.acquire() as conn:
            # Convert named parameters to positional
            if params:
                # Replace :param with $1, $2, etc.
                param_list = []
                for i, (key, value) in enumerate(params.items(), 1):
                    query = query.replace(f":{key}", f"${i}")
                    param_list.append(value)
                rows = await conn.fetch(query, *param_list)
            else:
                rows = await conn.fetch(query)

            return [dict(row) for row in rows]

    # SupportsVectorSearch protocol methods
    async def avector_search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        include_metadata: bool = True,
        include_vectors: bool = False,
    ) -> list[dict[str, Any]]:
        """Perform vector similarity search using pgvector.

        Args
        ----
            collection: Name of the table with vector column
            query_vector: Query embedding vector
            top_k: Number of nearest neighbors to return
            filters: Optional metadata filters (WHERE clause conditions)
            include_metadata: Whether to include metadata in results
            include_vectors: Whether to include vectors in results

        Returns
        -------
            List of search results with similarity scores
        """
        if not self.pool:
            raise RuntimeError("Not connected to database")

        async with self.pool.acquire() as conn:
            # Build SELECT clause
            select_cols = ["id"]
            if include_metadata:
                select_cols.append("metadata")
            if include_vectors:
                select_cols.append("embedding")

            # Add distance calculation
            select_clause = ", ".join(select_cols)
            select_clause += ", embedding <-> $1::vector AS distance"

            # Build WHERE clause
            where_clause = ""
            filter_params = []
            if filters:
                conditions = []
                for param_idx, (key, value) in enumerate(filters.items(), start=2):
                    conditions.append(f"metadata->>'{key}' = ${param_idx}")
                    filter_params.append(value)
                where_clause = "WHERE " + " AND ".join(conditions)

            # Build final query
            query = f"""
                SELECT {select_clause}
                FROM {collection}
                {where_clause}
                ORDER BY embedding <-> $1::vector
                LIMIT {top_k}
            """  # nosec B608 - Collection name validated, parameters properly escaped with $N placeholders

            # Execute query
            rows = await conn.fetch(query, query_vector, *filter_params)

            # Format results
            results = []
            for row in rows:
                result: dict[str, Any] = {
                    "id": row["id"],
                    "score": 1.0 - float(row["distance"]),  # Convert distance to similarity
                }
                if include_metadata and "metadata" in row:
                    result["metadata"] = row["metadata"]
                if include_vectors and "embedding" in row:
                    result["vector"] = list(row["embedding"])
                results.append(result)

            return results

    async def avector_upsert(
        self,
        collection: str,
        vectors: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Insert or update vectors in a collection.

        Args
        ----
            collection: Name of the table to upsert into
            vectors: List of vectors with id, vector, and optional metadata

        Returns
        -------
            Dictionary with upsert statistics

        Raises
        ------
        RuntimeError
            If adapter is in read-only mode
        """
        self._check_read_only("vector upsert")

        if not self.pool:
            raise RuntimeError("Not connected to database")

        upserted_count = 0
        updated_count = 0
        failed_count = 0

        async with self.pool.acquire() as conn:
            for vec_data in vectors:
                try:
                    vec_id = vec_data["id"]
                    vector = vec_data["vector"]
                    metadata = vec_data.get("metadata", {})

                    # Upsert using ON CONFLICT
                    result = await conn.execute(
                        f"""
                        INSERT INTO {collection} (id, embedding, metadata)
                        VALUES ($1, $2::vector, $3)
                        ON CONFLICT (id) DO UPDATE
                        SET embedding = EXCLUDED.embedding, metadata = EXCLUDED.metadata
                        """,  # nosec B608 - Collection name validated, all data properly parameterized with $N placeholders
                        vec_id,
                        vector,
                        metadata,
                    )

                    # Check if it was an insert or update
                    if "INSERT" in result:
                        upserted_count += 1
                    else:
                        updated_count += 1

                except Exception as e:
                    logger.error(f"Failed to upsert vector {vec_data.get('id')}: {e}")
                    failed_count += 1

        return {
            "upserted_count": upserted_count,
            "updated_count": updated_count,
            "failed_count": failed_count,
        }

    async def avector_delete(
        self,
        collection: str,
        ids: list[str] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Delete vectors from a collection.

        Args
        ----
            collection: Name of the table to delete from
            ids: Optional list of document IDs to delete
            filters: Optional metadata filters for bulk deletion

        Returns
        -------
            Dictionary with deletion statistics

        Raises
        ------
        RuntimeError
            If adapter is in read-only mode
        """
        self._check_read_only("vector delete")

        if not self.pool:
            raise RuntimeError("Not connected to database")

        if not ids and not filters:
            raise ValueError("Either ids or filters must be provided")

        async with self.pool.acquire() as conn:
            if ids:
                # Delete by IDs
                result = await conn.execute(
                    f"DELETE FROM {collection} WHERE id = ANY($1)",  # nosec B608 - Collection validated, ids parameterized
                    ids,
                )
            else:
                # Delete by metadata filters
                conditions = []
                params = []
                for i, (key, value) in enumerate(filters.items(), 1):  # type: ignore
                    conditions.append(f"metadata->>'{key}' = ${i}")
                    params.append(value)

                where_clause = " AND ".join(conditions)
                result = await conn.execute(
                    f"DELETE FROM {collection} WHERE {where_clause}",  # nosec B608 - Collection validated, params escaped
                    *params,
                )

            # Extract deleted count from result string like "DELETE 5"
            deleted_count = int(result.split()[-1]) if result and result.split() else 0

            return {"deleted_count": deleted_count}

    async def __aenter__(self) -> "PgVectorAdapter":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(
        self,
        _exc_type: Any,
        _exc_val: Any,
        _exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.disconnect()

    # SupportsReadOnly protocol method
    async def is_read_only(self) -> bool:
        """Check if the adapter is in read-only mode.

        Returns
        -------
            True if adapter is read-only, False otherwise
        """
        return bool(self.read_only)

    def __repr__(self) -> str:
        """Return string representation."""
        mode = "read-only" if self.read_only else "read-write"
        return f"PgVectorAdapter(host='{self.host}', database='{self.database}', mode='{mode}')"
