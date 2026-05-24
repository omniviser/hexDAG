"""Database adapters for HexDAG."""

from hexdag.stdlib.adapters.database.sqlite.sqlite_adapter import SQLiteAdapter

__all__ = ["SQLiteAdapter"]

try:
    from hexdag.stdlib.adapters.database.pgvector.pgvector_adapter import PgVectorAdapter
except ImportError:
    pass
else:
    __all__ += ["PgVectorAdapter"]
