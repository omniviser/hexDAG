"""PostgreSQL pgvector adapter using SQLAlchemy.

Production-ready vector store built on SQLAlchemy's battle-tested connection pool
and the official pgvector Python library.
"""

import time
from typing import Any

from pydantic import ConfigDict, SecretStr, field_validator
from sqlalchemy import Column, Integer, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base

from hexdag.core.configurable import AdapterConfig, ConfigurableAdapter, SecretField
from hexdag.core.ports.healthcheck import HealthStatus
from hexdag.core.registry.decorators import adapter
from hexdag.core.utils.sql_validation import validate_sql_identifier

try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    Vector = None  # type: ignore[assignment,misc]

from hexdag_plugins.storage.ports import VectorStorePort

Base = declarative_base()


class PgVectorConfig(AdapterConfig):
    """Configuration for PgVector adapter with SQLAlchemy.

    Attributes
    ----------
    connection_string : SecretStr | None
        PostgreSQL connection string with asyncpg driver
        (e.g., "postgresql+asyncpg://user:pass@localhost/db")
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
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    connection_string: SecretStr | None = SecretField(
        env_var="PGVECTOR_CONNECTION_STRING",
        description="PostgreSQL connection string (postgresql+asyncpg://...)",
    )
    table_name: str = "document_embeddings"
    embedding_dim: int = 384
    max_results: int = 5
    distance_metric: str = "cosine"

    # SQLAlchemy pool configuration
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: float = 30.0
    pool_recycle: int = 3600
    pool_pre_ping: bool = True

    @field_validator("table_name")
    @classmethod
    def validate_table_name(cls, v: str) -> str:
        """Validate table name to prevent SQL injection."""
        validate_sql_identifier(v, identifier_type="table", raise_on_invalid=True)
        if len(v) > 63:
            msg = f"Table name '{v}' exceeds PostgreSQL limit of 63 characters"
            raise ValueError(msg)
        return v

    @field_validator("distance_metric")
    @classmethod
    def validate_distance_metric(cls, v: str) -> str:
        """Validate distance metric."""
        valid_metrics = {"cosine", "l2", "inner_product"}
        if v not in valid_metrics:
            msg = f"Invalid distance metric: '{v}'. Must be one of: {', '.join(valid_metrics)}"
            raise ValueError(msg)
        return v


@adapter("vector_store", name="pgvector", namespace="plugin")
class PgVectorAdapter(ConfigurableAdapter, VectorStorePort):
    """PostgreSQL pgvector adapter using SQLAlchemy.

    Built on SQLAlchemy's async engine with connection pooling for production use.
    Uses the official pgvector library for vector operations.

    Benefits
    --------
    - SQLAlchemy's mature connection pooling (5-20 connections)
    - Automatic connection health checks (pool_pre_ping)
    - Connection recycling to prevent stale connections
    - Type-safe ORM with pgvector support
    - Production-ready error handling

    Examples
    --------
    Basic usage with connection pooling::

        from hexdag.core.registry import registry

        pgvector = registry.get("pgvector", namespace="plugin")(
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

    Configuration options::

        pgvector = registry.get("pgvector", namespace="plugin")(
            connection_string="postgresql+asyncpg://localhost/mydb",
            table_name="embeddings",
            embedding_dim=1536,
            pool_size=20,           # Connection pool size
            max_overflow=10,        # Extra connections if needed
            pool_recycle=3600,      # Recycle after 1 hour
            pool_pre_ping=True,     # Check connection health
            distance_metric="cosine"
        )
    """

    Config = PgVectorConfig

    def __init__(self, **kwargs: Any) -> None:
        """Initialize PgVector adapter with SQLAlchemy engine."""
        if Vector is None:
            msg = (
                "pgvector is required for PgVector adapter. "
                "Install it with: uv pip install pgvector"
            )
            raise ImportError(msg)

        super().__init__(**kwargs)
        self._engine: AsyncEngine | None = None
        self._model_class = None
        self._initialized = False

    def _create_model(self):
        """Create SQLAlchemy model dynamically based on config."""
        table_name = self.config.table_name
        embedding_dim = self.config.embedding_dim

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
        if self.config.connection_string is None:
            msg = "connection_string is required for PgVector adapter"
            raise ValueError(msg)

        # Create async engine with connection pool
        self._engine = create_async_engine(
            self.config.connection_string.get_secret_value(),
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            pool_timeout=self.config.pool_timeout,
            pool_recycle=self.config.pool_recycle,
            pool_pre_ping=self.config.pool_pre_ping,
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
            index_name = f"{self.config.table_name}_embedding_idx"

            # Check if index exists
            index_exists = await conn.scalar(
                text("SELECT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = :index_name)"),
                {"index_name": index_name},
            )

            if not index_exists:
                await conn.execute(
                    text(f"""
                        CREATE INDEX {index_name}
                        ON {self.config.table_name}
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

        k = top_k or self.config.max_results

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
                similarity = (
                    1 - distance if self.config.distance_metric in ["cosine", "l2"] else distance
                )

                formatted_results.append({
                    "id": record.id,
                    "text": record.text,
                    "metadata": record.metadata or {},
                    "similarity_score": float(similarity),
                })

        return formatted_results

    async def aclear(self) -> dict[str, Any]:
        """Clear all documents from the table."""
        if not self._initialized:
            await self.asetup()

        async with AsyncSession(self._engine) as session:
            # Count before deletion
            count = await session.scalar(text(f"SELECT COUNT(*) FROM {self.config.table_name}"))

            # Truncate table
            await session.execute(text(f"TRUNCATE TABLE {self.config.table_name}"))
            await session.commit()

        return {"removed": count}

    async def acount(self) -> int:
        """Get the number of documents in the vector store."""
        if not self._initialized:
            await self.asetup()

        async with AsyncSession(self._engine) as session:
            count = await session.scalar(text(f"SELECT COUNT(*) FROM {self.config.table_name}"))
            return count or 0

    async def aget_stats(self) -> dict[str, Any]:
        """Get statistics about the vector store and connection pool."""
        if not self._initialized:
            await self.asetup()

        async with AsyncSession(self._engine) as session:
            # Get document count
            count = await session.scalar(text(f"SELECT COUNT(*) FROM {self.config.table_name}"))

            # Get table size
            size = await session.scalar(
                text(f"SELECT pg_size_pretty(pg_total_relation_size('{self.config.table_name}'))")
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
                    "max_overflow": self.config.max_overflow,
                }

        return {
            "document_count": count,
            "table_name": self.config.table_name,
            "table_size": size,
            "embedding_dim": self.config.embedding_dim,
            "distance_metric": self.config.distance_metric,
            "connection_pool": pool_stats,
        }

    def _get_distance_operator(self) -> str:
        """Get PostgreSQL distance operator for the configured metric."""
        operators = {
            "cosine": "<=>",
            "l2": "<->",
            "inner_product": "<#>",
        }
        return operators.get(self.config.distance_metric, "<=>")

    def _get_distance_function(self):
        """Get pgvector distance function for SQLAlchemy."""
        from pgvector.sqlalchemy import Vector

        metric_map = {
            "cosine": "cosine_distance",
            "l2": "l2_distance",
            "inner_product": "max_inner_product",
        }
        func_name = metric_map.get(self.config.distance_metric, "cosine_distance")
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
                    {"table_name": self.config.table_name},
                )

                latency_ms = (time.time() - start_time) * 1000

                if not table_exists:
                    return HealthStatus(
                        status="degraded",
                        adapter_name="pgvector",
                        port_name="vector_store",
                        latency_ms=latency_ms,
                        details={
                            "table": self.config.table_name,
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
                        "table": self.config.table_name,
                        "document_count": count,
                        "embedding_dim": self.config.embedding_dim,
                        "distance_metric": self.config.distance_metric,
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
                    "table": self.config.table_name,
                    "error_type": type(e).__name__,
                },
            )

    def __repr__(self) -> str:
        """String representation."""
        pool_info = ""
        if self._engine and self._engine.pool:
            pool = self._engine.pool
            pool_info = f", pool={pool.checkedout()}/{pool.size()}"
        return f"PgVectorAdapter(table={self.config.table_name}{pool_info})"
