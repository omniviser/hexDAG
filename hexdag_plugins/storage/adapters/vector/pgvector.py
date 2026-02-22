"""PostgreSQL pgvector adapter using SQLAlchemy.

Production-ready vector store built on SQLAlchemy's battle-tested connection pool
and the official pgvector Python library.
"""

import os
import time
from typing import Any

from hexdag.kernel.ports.healthcheck import HealthStatus
from hexdag.kernel.utils.sql_validation import validate_sql_identifier
from sqlalchemy import Column, Integer, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base

try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    Vector = None  # type: ignore[assignment,misc]

from hexdag_plugins.storage.ports import VectorStorePort

Base = declarative_base()


class PgVectorAdapter(VectorStorePort):
    """PostgreSQL pgvector adapter using SQLAlchemy.

    Built on SQLAlchemy's async engine with connection pooling for production use.
    Uses the official pgvector library for vector operations.

    Parameters
    ----------
    connection_string : str | None
        PostgreSQL connection string with asyncpg driver
        (e.g., "postgresql+asyncpg://user:pass@localhost/db")
        Auto-resolved from PGVECTOR_CONNECTION_STRING env var if not provided
    table_name : str
        Name of the vector table (default: "document_embeddings")
    embedding_dim : int
        Dimension of embedding vectors (default: 384)
    max_results : int
        Maximum number of search results (default: 5)
    distance_metric : str
        Distance metric: "cosine", "l2", or "inner_product" (default: "cosine")
    pool_size : int
        SQLAlchemy connection pool size (default: 5)
    max_overflow : int
        Maximum overflow connections beyond pool_size (default: 10)
    pool_timeout : float
        Timeout for getting connection from pool (default: 30.0)
    pool_recycle : int
        Recycle connections after N seconds (default: 3600 - 1 hour)
    pool_pre_ping : bool
        Test connections before using them (default: True)

    Examples
    --------
    Basic usage with connection pooling::

        pgvector = PgVectorAdapter(
            connection_string="postgresql+asyncpg://localhost/mydb",
            pool_size=10,
            max_overflow=20
        )
        await pgvector.asetup()

        # Add documents
        await pgvector.aadd_documents(documents, embeddings)

        # Search with vector similarity
        results = await pgvector.asearch(
            "query text",
            query_embedding=embedding,
            top_k=5
        )
    """

    _hexdag_icon = "Boxes"
    _hexdag_color = "#336791"  # PostgreSQL blue

    def __init__(
        self,
        connection_string: str | None = None,
        table_name: str = "document_embeddings",
        embedding_dim: int = 384,
        max_results: int = 5,
        distance_metric: str = "cosine",
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: float = 30.0,
        pool_recycle: int = 3600,
        pool_pre_ping: bool = True,
    ) -> None:
        """Initialize PgVector adapter with SQLAlchemy engine.

        Parameters
        ----------
        connection_string : str | None
            PostgreSQL connection string (auto-resolved from PGVECTOR_CONNECTION_STRING env var)
        table_name : str
            Name of the vector table (default: "document_embeddings")
        embedding_dim : int
            Dimension of embedding vectors (default: 384)
        max_results : int
            Maximum number of search results (default: 5)
        distance_metric : str
            Distance metric: "cosine", "l2", or "inner_product" (default: "cosine")
        pool_size : int
            SQLAlchemy connection pool size (default: 5)
        max_overflow : int
            Maximum overflow connections beyond pool_size (default: 10)
        pool_timeout : float
            Timeout for getting connection from pool (default: 30.0)
        pool_recycle : int
            Recycle connections after N seconds (default: 3600)
        pool_pre_ping : bool
            Test connections before using them (default: True)
        """
        if Vector is None:
            msg = (
                "pgvector is required for PgVector adapter. "
                "Install it with: uv pip install pgvector"
            )
            raise ImportError(msg)

        # Validate inputs
        validate_sql_identifier(table_name, identifier_type="table", raise_on_invalid=True)
        if len(table_name) > 63:
            msg = f"Table name '{table_name}' exceeds PostgreSQL limit of 63 characters"
            raise ValueError(msg)

        valid_metrics = {"cosine", "l2", "inner_product"}
        if distance_metric not in valid_metrics:
            msg = (
                f"Invalid distance metric: '{distance_metric}'. "
                f"Must be one of: {', '.join(valid_metrics)}"
            )
            raise ValueError(msg)

        # Resolve connection string from env if not provided
        self._connection_string = connection_string or os.getenv("PGVECTOR_CONNECTION_STRING")
        self._table_name = table_name
        self._embedding_dim = embedding_dim
        self._max_results = max_results
        self._distance_metric = distance_metric
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._pool_timeout = pool_timeout
        self._pool_recycle = pool_recycle
        self._pool_pre_ping = pool_pre_ping

        self._engine: AsyncEngine | None = None
        self._model_class = None
        self._initialized = False

    def _create_model(self):
        """Create SQLAlchemy model dynamically based on config."""
        table_name = self._table_name
        embedding_dim = self._embedding_dim

        class EmbeddingModel(Base):
            __tablename__ = table_name
            __table_args__ = {"extend_existing": True}

            id = Column(Integer, primary_key=True)
            text = Column(Text, nullable=False)
            embedding = Column(Vector(embedding_dim))
            metadata = Column(JSONB, default={})

        return EmbeddingModel

    async def asetup(self) -> None:
        """Initialize SQLAlchemy engine and create tables."""
        if self._connection_string is None:
            msg = "connection_string is required for PgVector adapter"
            raise ValueError(msg)

        # Create async engine with connection pool
        self._engine = create_async_engine(
            self._connection_string,
            pool_size=self._pool_size,
            max_overflow=self._max_overflow,
            pool_timeout=self._pool_timeout,
            pool_recycle=self._pool_recycle,
            pool_pre_ping=self._pool_pre_ping,
            echo=False,  # Set to True for SQL debugging
        )

        # Create dynamic model
        self._model_class = self._create_model()

        # Enable pgvector extension and create tables
        async with self._engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)

            # Create vector similarity index
            distance_op = self._get_distance_operator()
            index_name = f"{self._table_name}_embedding_idx"

            # Check if index exists
            index_exists = await conn.scalar(
                text("SELECT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = :index_name)"),
                {"index_name": index_name},
            )

            if not index_exists:
                await conn.execute(
                    text(f"""
                        CREATE INDEX {index_name}
                        ON {self._table_name}
                        USING ivfflat (embedding {distance_op})
                        WITH (lists = 100)
                    """)
                )

        self._initialized = True

    async def aclose(self) -> None:
        """Close SQLAlchemy engine and connection pool."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._initialized = False

    async def aadd_documents(
        self,
        documents: list[dict[str, Any]],
        embeddings: list[list[float]] | None = None,
    ) -> dict[str, Any]:
        """Add documents with embeddings to the vector store."""
        if not self._initialized:
            await self.asetup()

        if embeddings is None:
            msg = "Embeddings are required for PgVector adapter. Example: embeddings = \
                await embedding_adapter.aembed_texts([doc['text'] for doc in documents])"
            raise ValueError(msg)

        if len(documents) != len(embeddings):
            msg = f"Mismatch: {len(documents)} documents but {len(embeddings)} embeddings"
            raise ValueError(msg)

        inserted_ids = []
        async with AsyncSession(self._engine) as session:
            for doc, embedding in zip(documents, embeddings, strict=False):
                record = self._model_class(
                    text=doc.get("text", ""),
                    embedding=embedding,
                    metadata=doc.get("metadata", {}),
                )
                session.add(record)

            await session.commit()

            # Get IDs of inserted records (need to refresh to get IDs)
            inserted_ids.extend(record.id for record in session.new if hasattr(record, "id"))

        return {
            "added": len(documents),
            "ids": inserted_ids,
        }

    async def asearch(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        top_k: int | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar documents using vector similarity."""
        if not self._initialized:
            await self.asetup()

        if query_embedding is None:
            msg = (
                "Query embedding is required for PgVector search. "
                "Example: query_embedding = await embedding_adapter.aembed_text(query)"
            )
            raise ValueError(msg)

        k = top_k or self._max_results

        # Build query with distance function
        distance_func = self._get_distance_function()

        async with AsyncSession(self._engine) as session:
            # Build base query
            query_obj = session.query(self._model_class).order_by(
                distance_func(self._model_class.embedding, query_embedding)
            )

            # Add metadata filtering if provided
            if filter_metadata:
                for key, value in filter_metadata.items():
                    query_obj = query_obj.filter(
                        self._model_class.metadata[key].astext == str(value)
                    )

            # Execute query
            query_obj = query_obj.limit(k)
            results = (await session.execute(query_obj)).scalars().all()

            # Format results
            formatted_results = []
            for record in results:
                # Calculate similarity score
                distance = distance_func(record.embedding, query_embedding)
                similarity = 1 - distance if self._distance_metric in ["cosine", "l2"] else distance

                formatted_results.append(
                    {
                        "id": record.id,
                        "text": record.text,
                        "metadata": record.metadata or {},
                        "similarity_score": float(similarity),
                    }
                )

        return formatted_results

    async def aclear(self) -> dict[str, Any]:
        """Clear all documents from the table."""
        if not self._initialized:
            await self.asetup()

        async with AsyncSession(self._engine) as session:
            # Count before deletion
            count = await session.scalar(text(f"SELECT COUNT(*) FROM {self._table_name}"))

            # Truncate table
            await session.execute(text(f"TRUNCATE TABLE {self._table_name}"))
            await session.commit()

        return {"removed": count}

    async def acount(self) -> int:
        """Get the number of documents in the vector store."""
        if not self._initialized:
            await self.asetup()

        async with AsyncSession(self._engine) as session:
            count = await session.scalar(text(f"SELECT COUNT(*) FROM {self._table_name}"))
            return count or 0

    async def aget_stats(self) -> dict[str, Any]:
        """Get statistics about the vector store and connection pool."""
        if not self._initialized:
            await self.asetup()

        async with AsyncSession(self._engine) as session:
            # Get document count
            count = await session.scalar(text(f"SELECT COUNT(*) FROM {self._table_name}"))

            # Get table size
            size = await session.scalar(
                text(f"SELECT pg_size_pretty(pg_total_relation_size('{self._table_name}'))")
            )

            # Get pool statistics
            pool_stats = {}
            if self._engine and self._engine.pool:
                pool = self._engine.pool
                pool_stats = {
                    "pool_size": pool.size(),
                    "checked_in": pool.checkedin(),
                    "checked_out": pool.checkedout(),
                    "overflow": pool.overflow(),
                    "max_overflow": self._max_overflow,
                }

        return {
            "document_count": count,
            "table_name": self._table_name,
            "table_size": size,
            "embedding_dim": self._embedding_dim,
            "distance_metric": self._distance_metric,
            "connection_pool": pool_stats,
        }

    def _get_distance_operator(self) -> str:
        """Get PostgreSQL distance operator for the configured metric."""
        operators = {
            "cosine": "<=>",
            "l2": "<->",
            "inner_product": "<#>",
        }
        return operators.get(self._distance_metric, "<=>")

    def _get_distance_function(self):
        """Get pgvector distance function for SQLAlchemy."""
        from pgvector.sqlalchemy import Vector

        metric_map = {
            "cosine": "cosine_distance",
            "l2": "l2_distance",
            "inner_product": "max_inner_product",
        }
        func_name = metric_map.get(self._distance_metric, "cosine_distance")
        return getattr(Vector, func_name, Vector.cosine_distance)

    async def ahealth_check(self) -> HealthStatus:
        """Check PostgreSQL, pgvector, and connection pool health."""
        start_time = time.time()

        try:
            if not self._initialized:
                await self.asetup()

            async with AsyncSession(self._engine) as session:
                # Test basic connectivity
                await session.execute(text("SELECT 1"))

                # Test pgvector extension
                await session.execute(text("SELECT vector_dims(vector('[1,2,3]'))"))

                # Test table exists
                table_exists = await session.scalar(
                    text(
                        """
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables
                            WHERE table_name = :table_name
                        )
                        """
                    ),
                    {"table_name": self._table_name},
                )

                latency_ms = (time.time() - start_time) * 1000

                if not table_exists:
                    return HealthStatus(
                        status="degraded",
                        adapter_name="pgvector",
                        port_name="vector_store",
                        latency_ms=latency_ms,
                        details={
                            "table": self._table_name,
                            "message": "Table does not exist (not yet initialized)",
                        },
                    )

                # Get document count
                count = await self.acount()

                # Get pool statistics
                pool_stats = {}
                if self._engine and self._engine.pool:
                    pool = self._engine.pool
                    pool_stats = {
                        "pool_size": pool.size(),
                        "checked_in": pool.checkedin(),
                        "checked_out": pool.checkedout(),
                    }

                return HealthStatus(
                    status="healthy",
                    adapter_name="pgvector",
                    port_name="vector_store",
                    latency_ms=latency_ms,
                    details={
                        "table": self._table_name,
                        "document_count": count,
                        "embedding_dim": self._embedding_dim,
                        "distance_metric": self._distance_metric,
                        "connection_pool": pool_stats,
                        "sqlalchemy_version": "async",
                    },
                )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthStatus(
                status="unhealthy",
                adapter_name="pgvector",
                port_name="vector_store",
                error=e,
                latency_ms=latency_ms,
                details={
                    "table": self._table_name,
                    "error_type": type(e).__name__,
                },
            )

    def __repr__(self) -> str:
        """String representation."""
        pool_info = ""
        if self._engine and self._engine.pool:
            pool = self._engine.pool
            pool_info = f", pool={pool.checkedout()}/{pool.size()}"
        return f"PgVectorAdapter(table={self._table_name}{pool_info})"
