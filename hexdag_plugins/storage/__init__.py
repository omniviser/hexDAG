"""hexdag-storage: Low-level storage infrastructure for hexDAG.

Provides database, vector store, and file storage adapters with
connection pooling, health checks, and async-first operations.
"""

from .file import LocalFileStorage
from .vector import ChromaDBAdapter, InMemoryVectorStore, PgVectorAdapter

__version__ = "0.1.0"

__all__ = [
    # Vector Stores
    "InMemoryVectorStore",
    "PgVectorAdapter",
    "ChromaDBAdapter",
    # File Storage
    "LocalFileStorage",
]
