"""Lightweight caching utilities for hot-path optimizations.

Provides reusable cache primitives for deterministic computations that are
called repeatedly with the same inputs (e.g., Pydantic model creation,
schema generation, tool instruction building).

Examples
--------
Module-level keyed cache::

    from hexdag.kernel.utils.caching import KeyedCache

    _model_cache: KeyedCache[type] = KeyedCache()
    model = _model_cache.get_or_create(("UserInput", schema_key), factory_fn)

Class-keyed cache (for Pydantic model classes)::

    _schema_cache: KeyedCache[str] = KeyedCache()
    instruction = _schema_cache.get_or_create(OutputModel, build_instruction)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


class KeyedCache[V]:
    """Simple dict-backed cache with get-or-create semantics.

    Thread-safe for CPython (dict operations are atomic under GIL).
    No max-size eviction — intended for bounded key spaces
    (e.g., unique schemas in a pipeline, model classes).

    Examples
    --------
    >>> cache: KeyedCache[int] = KeyedCache()
    >>> cache.get_or_create("key", lambda: 42)
    42
    >>> cache.get_or_create("key", lambda: 99)  # cached
    42
    >>> len(cache)
    1
    """

    __slots__ = ("_store",)

    def __init__(self) -> None:
        self._store: dict[Any, V] = {}

    def get_or_create(self, key: Any, factory: Callable[[], V]) -> V:
        """Return cached value or create via factory and cache it.

        Parameters
        ----------
        key : Any
            Hashable cache key.
        factory : Callable[[], V]
            Zero-argument callable that creates the value on cache miss.
        """
        try:
            return self._store[key]
        except KeyError:
            value = factory()
            self._store[key] = value
            return value

    def get(self, key: Any) -> V | None:
        """Return cached value or None."""
        return self._store.get(key)

    def put(self, key: Any, value: V) -> None:
        """Explicitly store a value."""
        self._store[key] = value

    def clear(self) -> None:
        """Remove all cached entries."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: Any) -> bool:
        return key in self._store


def schema_cache_key(schema: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Create a hashable cache key from a schema dict.

    Sorts items and uses ``repr()`` for values to handle type objects,
    strings, and tuples uniformly.

    Examples
    --------
    >>> schema_cache_key({"name": str, "age": int})  # doctest: +SKIP
    (('age', "<class 'int'>"), ('name', "<class 'str'>"))
    """
    return tuple(sorted((k, repr(v)) for k, v in schema.items()))
