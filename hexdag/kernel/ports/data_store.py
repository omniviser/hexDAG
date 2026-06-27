"""Unified DataStore port — capability-based storage abstraction.

DataStore unifies Memory (key-value) and Database (query) ports under
a single capability-based protocol hierarchy.  Adapters implement only
the sub-protocols they support; callers check capabilities at runtime
with ``isinstance(store, SupportsXxx)``.

Capability matrix
-----------------
+---------------------+--------------------------------------------+
| Sub-protocol        | Description                                |
+=====================+============================================+
| SupportsKeyValue    | Core get/set/delete/exists/list_keys       |
| SupportsTTL         | Key-value with time-to-live                |
| SupportsQuery       | SQL-style parameterised queries            |
| SupportsSchema      | Table schema + relationship inspection     |
+---------------------+--------------------------------------------+

Existing optional protocols (SupportsVectorSearch, SupportsStreamingQuery,
SupportsRawSQL, SupportsReadOnly, SupportsTransactions) are orthogonal
and can be mixed freely.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DataStore(Protocol):
    """Marker protocol for unified storage adapters.

    ``DataStore`` groups the capability sub-protocols (``SupportsKeyValue``,
    ``SupportsTTL``, ``SupportsQuery``, ``SupportsSchema``,
    ``SupportsTransactions``) under a single identity — like ``LLM`` groups
    its capabilities.

    Adapters implement ``DataStore`` plus whichever capabilities they
    support::

        class SQLiteStore(DataStore, SupportsKeyValue, SupportsQuery):
            ...

    No methods are required on ``DataStore`` itself; capabilities are
    checked at runtime with ``isinstance(store, SupportsXxx)``.
    """

    ...


# ---------------------------------------------------------------------------
# Key-value capabilities (supersedes Memory port)
# ---------------------------------------------------------------------------


@runtime_checkable
class SupportsKeyValue(Protocol):
    """Core key-value storage — the minimal DataStore capability.

    Every adapter that was previously a Memory adapter should implement
    this protocol.
    """

    @abstractmethod
    async def aget(self, key: str) -> Any:
        """Retrieve a value by key.  Returns ``None`` when key is absent."""
        ...

    @abstractmethod
    async def aset(self, key: str, value: Any) -> None:
        """Store *value* under *key* (upsert semantics)."""
        ...

    @abstractmethod
    async def adelete(self, key: str) -> bool:
        """Delete *key*.  Returns ``True`` if the key existed."""
        ...

    @abstractmethod
    async def aexists(self, key: str) -> bool:
        """Check whether *key* exists in the store."""
        ...

    @abstractmethod
    async def alist_keys(self, prefix: str = "") -> list[str]:
        """List keys, optionally filtered by *prefix*."""
        ...


@runtime_checkable
class SupportsTTL(Protocol):
    """Key-value storage with time-to-live support."""

    @abstractmethod
    async def aset_with_ttl(self, key: str, value: Any, ttl_seconds: int) -> None:
        """Store *value* under *key* with automatic expiry after *ttl_seconds*."""
        ...


# ---------------------------------------------------------------------------
# Query capabilities (supersedes DatabasePort)
# ---------------------------------------------------------------------------


@runtime_checkable
class SupportsQuery(Protocol):
    """SQL-style parameterised query execution."""

    @abstractmethod
    async def aexecute_query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute *query* with optional *params* and return result rows."""
        ...


@runtime_checkable
class SupportsSchema(Protocol):
    """Database schema introspection."""

    @abstractmethod
    async def aget_table_schemas(self) -> dict[str, dict[str, Any]]:
        """Return schema information for every table.

        The returned dict maps table names to schema dicts with at least
        ``columns``, ``primary_keys``, and ``foreign_keys`` entries.
        """
        ...

    @abstractmethod
    async def aget_relationships(self) -> list[dict[str, Any]]:
        """Return foreign-key relationships across all tables."""
        ...

    @abstractmethod
    async def aget_indexes(self, table: str) -> list[dict[str, Any]]:
        """Return index definitions for *table*."""
        ...

    @abstractmethod
    async def aget_table_statistics(self, table: str) -> dict[str, Any]:
        """Return statistics (row count, approximate size, …) for *table*."""
        ...


# ---------------------------------------------------------------------------
# Transaction capability — plugin-owned (see hexdag_plugins.database._ports)
# ---------------------------------------------------------------------------
#
# ``SupportsTransactions`` and ``SupportsSessionFactory`` are transaction
# contracts only the database plugin and its users consume — the kernel and
# stdlib never reference them.  They now live in the database plugin alongside
# their implementation.  For back-compat they remain importable from here via
# the lazy resolver below (PEP 562), backed by the kernel port registry.


def __getattr__(name: str) -> Any:
    """Resolve plugin-contributed port protocols lazily (PEP 562).

    Keeps ``from hexdag.kernel.ports.data_store import SupportsTransactions``
    working after the protocols moved to the database plugin.
    """
    if name in ("SupportsTransactions", "SupportsSessionFactory"):
        from hexdag.kernel.ports.registry import resolve_plugin_port

        proto = resolve_plugin_port(name)
        if proto is not None:
            return proto
        raise AttributeError(
            f"{name!r} is provided by the database plugin; "
            f"install it with `pip install hexdag-plugins[database]`"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Collection storage capability (lib persistence)
# ---------------------------------------------------------------------------


@runtime_checkable
class SupportsCollectionStorage(Protocol):
    """Collection-scoped document storage for lib persistence.

    Provides CRUD operations scoped by collection name, where each
    record is a ``dict[str, Any]`` identified by a string key.

    Libs (ProcessRegistry, EntityState, Scheduler) use this protocol
    to optionally persist their state to a pluggable backend.  When no
    backend is provided they fall back to in-memory dicts.

    Example::

        class SQLCollectionStorage:
            async def asave(self, collection: str, key: str, data: dict[str, Any]) -> None:
                ...
    """

    @abstractmethod
    async def asave(self, collection: str, key: str, data: dict[str, Any]) -> None:
        """Save a document to a collection (upsert semantics).

        Args
        ----
            collection: The collection name (e.g. ``"pipeline_runs"``).
            key: Unique identifier within the collection.
            data: The document to store.
        """
        ...

    @abstractmethod
    async def aload(self, collection: str, key: str) -> dict[str, Any] | None:
        """Load a document by key.  Returns ``None`` if not found.

        Args
        ----
            collection: The collection name.
            key: Unique identifier within the collection.
        """
        ...

    @abstractmethod
    async def aquery(
        self, collection: str, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Query documents in a collection, optionally filtered.

        Filters use exact equality matching on document fields.

        Args
        ----
            collection: The collection name.
            filters: Optional dict of field-name → value equality filters.
        """
        ...

    @abstractmethod
    async def adelete(self, collection: str, key: str) -> bool:
        """Delete a document.  Returns ``True`` if it existed.

        Args
        ----
            collection: The collection name.
            key: Unique identifier within the collection.
        """
        ...


__all__ = [
    "DataStore",
    "SupportsCollectionStorage",
    "SupportsKeyValue",
    "SupportsQuery",
    "SupportsSchema",
    "SupportsTTL",
    "SupportsTransactions",  # noqa: F822 # pyright: ignore[reportUnsupportedDunderAll]
]
