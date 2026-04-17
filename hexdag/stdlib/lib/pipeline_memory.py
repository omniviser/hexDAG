"""PipelineMemory — run-scoped key-value store for cross-node shared state.

Auto-registered for every pipeline run.  Provides a mutable context that
any node can read/write — useful for cross-cutting data that doesn't fit
the producer→consumer DAG pattern.

Access patterns:

- **Expressions:** ``memory('offer_history', [])``
- **Node functions:** ``get_pipeline_memory().set('key', value)``
- **Agent tools:** ``get_memory(key)``, ``set_memory(key, value)``

The store is scoped to a single pipeline run and discarded on completion.
Cross-run persistence is handled by the preload mechanism (see
``spec.memory.preload``).
"""

from __future__ import annotations

from typing import Any

from hexdag.kernel.service import Service, step, tool


class PipelineMemory(Service):
    """Run-scoped key-value store, auto-registered for every pipeline run.

    Exposed tools
    -------------
    - ``get(key, default?)`` — read a value
    - ``set(key, value)`` — write a value
    - ``update(data)`` — merge a dict into the store
    - ``snapshot()`` — return the entire store as a dict
    """

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Agent-callable tools + DAG step methods
    # ------------------------------------------------------------------

    @tool
    @step
    async def get(self, key: str, default: Any = None) -> Any:
        """Read a value from pipeline memory.

        Args
        ----
            key: The key to look up.
            default: Value to return if key is not found.

        Returns
        -------
            The stored value, or *default* if absent.
        """
        return self._store.get(key, default)

    @tool
    @step
    async def set(self, key: str, value: Any) -> dict[str, Any]:
        """Write a value to pipeline memory.

        Args
        ----
            key: The key to store under.
            value: The value to store.

        Returns
        -------
            Confirmation dict with key and stored status.
        """
        self._store[key] = value
        return {"key": key, "stored": True}

    @tool
    @step
    async def update(self, data: dict[str, Any]) -> dict[str, Any]:
        """Merge a dictionary into pipeline memory.

        Args
        ----
            data: Key-value pairs to merge.

        Returns
        -------
            Confirmation dict with count of keys updated.
        """
        self._store.update(data)
        return {"keys_updated": len(data)}

    @tool
    @step
    async def snapshot(self) -> dict[str, Any]:
        """Return the entire pipeline memory store.

        Returns
        -------
            A shallow copy of the full store.
        """
        return dict(self._store)
