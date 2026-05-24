"""pgvector adapter for PostgreSQL with vector similarity search."""

try:
    from hexdag.stdlib.adapters.database.pgvector.pgvector_adapter import PgVectorAdapter
except ImportError as _exc:
    raise ImportError(
        "PgVectorAdapter requires 'asyncpg'. Install with: pip install hexdag[pgvector]"
    ) from _exc

__all__ = ["PgVectorAdapter"]
