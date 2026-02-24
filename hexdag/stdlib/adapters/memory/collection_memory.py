"""In-memory implementation of SupportsCollectionStorage.

Provides a fast, zero-dependency collection storage adapter for testing
and as the default backend when no persistent storage is configured.

Usage::

    from hexdag.stdlib.adapters.memory.collection_memory import InMemoryCollectionStorage

    storage = InMemoryCollectionStorage()
    await storage.asave("pipeline_runs", "run-1", {"status": "running"})
    doc = await storage.aload("pipeline_runs", "run-1")
"""

from __future__ import annotations

from typing import Any


class InMemoryCollectionStorage:
    """In-memory ``SupportsCollectionStorage`` for testing and default behaviour.

    Data is stored in nested dicts: ``collection → key → data``.
    All operations are O(1) for save/load/delete and O(n) for query.
    """

    def __init__(self) -> None:
        """Initialise empty collection store."""
        self._data: dict[str, dict[str, dict[str, Any]]] = {}

    async def asave(self, collection: str, key: str, data: dict[str, Any]) -> None:
        """Save a document to a collection (upsert semantics)."""
        self._data.setdefault(collection, {})[key] = data

    async def aload(self, collection: str, key: str) -> dict[str, Any] | None:
        """Load a document by key.  Returns ``None`` if not found."""
        return self._data.get(collection, {}).get(key)

    async def aquery(
        self, collection: str, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Query documents with optional equality filters."""
        docs = list(self._data.get(collection, {}).values())
        if filters:
            docs = [d for d in docs if all(d.get(k) == v for k, v in filters.items())]
        return docs

    async def adelete(self, collection: str, key: str) -> bool:
        """Delete a document.  Returns ``True`` if it existed."""
        coll = self._data.get(collection, {})
        if key in coll:
            del coll[key]
            return True
        return False
