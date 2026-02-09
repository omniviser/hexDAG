"""Storage plugin for hexDAG framework.

Plugin Structure
----------------
::

    storage/
    ├── __init__.py          # This file - re-exports from adapters/
    ├── adapters/            # Adapter implementations
    │   ├── __init__.py
    │   ├── file/            # File storage adapters
    │   │   └── local.py     # LocalFileStorage
    │   ├── vector/          # Vector store adapters
    │   │   ├── in_memory.py # InMemoryVectorStore
    │   │   ├── pgvector.py  # PgVectorAdapter
    │   │   └── chromadb.py  # ChromaDBAdapter
    │   └── sql/             # SQL database adapters
    │       ├── mysql.py     # MySQLAdapter
    │       └── postgresql.py # PostgreSQLAdapter
    ├── ports/               # Custom port protocols
    │   └── vector_store.py  # VectorStorePort
    └── tests/               # Test files

Available Adapters
------------------
- ``LocalFileStorage``: Local filesystem storage (inherits ``FileStoragePort``)
- ``InMemoryVectorStore``: In-memory vector store (inherits ``VectorStorePort``)
- ``PgVectorAdapter``: PostgreSQL pgvector (inherits ``VectorStorePort``)
- ``ChromaDBAdapter``: ChromaDB vector store (inherits ``VectorStorePort``)
"""

from hexdag_plugins.storage.adapters import (
    ChromaDBAdapter,
    InMemoryVectorStore,
    LocalFileStorage,
    PgVectorAdapter,
)

__version__ = "0.1.0"

__all__ = [
    # File Storage
    "LocalFileStorage",
    # Vector Stores
    "InMemoryVectorStore",
    "PgVectorAdapter",
    "ChromaDBAdapter",
]
