"""Port protocols owned by the database plugin.

These transaction contracts used to live in ``hexdag.kernel.ports.data_store``
but they are only meaningful to the database plugin and its users — the kernel
and stdlib never consume them.  Following the precedent of
``hexdag_plugins/_ports.py`` (email / file storage), the plugin owns both the
contract and the implementation for its domain.

On import this module registers the protocols with the kernel port registry so
``from hexdag.kernel.ports import SupportsTransactions`` keeps resolving for
back-compat (see :mod:`hexdag.kernel.ports.registry`).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Protocol, runtime_checkable

from hexdag.kernel.ports.registry import register_port


@runtime_checkable
class SupportsTransactions(Protocol):
    """Adapter-owned transaction lifecycle.

    Transaction ownership is at the adapter level — the orchestrator never
    begins/commits/rolls-back transactions itself.
    """

    @abstractmethod
    async def abegin(self) -> None:
        """Begin a new transaction."""
        ...

    @abstractmethod
    async def acommit(self) -> None:
        """Commit the current transaction."""
        ...

    @abstractmethod
    async def arollback(self) -> None:
        """Roll back the current transaction."""
        ...


@runtime_checkable
class SupportsSessionFactory(Protocol):
    """Per-step session factory for saga-safe database access.

    Instead of sharing one session across all nodes, each ``@step`` call
    obtains its own session from the factory, commits independently, and
    releases the connection.  This eliminates shared-session coupling and
    enables the saga pattern (independent commit + compensation on failure).

    Usage in a service::

        class OrderService(Service):
            def __init__(self, db: SupportsSessionFactory) -> None:
                self._db = db

            @step
            async def update_order(self, order_id: str, status: str) -> dict:
                async with self._db.asession() as session:
                    order = await session.get(Order, order_id)
                    order.status = status
                    await session.commit()
                    return {"order_id": order_id, "status": status}
    """

    @abstractmethod
    def asession(self) -> Any:
        """Return an async context manager that yields a database session.

        Each call produces an independent session with its own transaction
        scope.  The session is committed or rolled back by the caller, and
        the underlying connection is returned to the pool on exit.
        """
        ...


# Register with the kernel so the names stay importable from
# ``hexdag.kernel.ports`` (back-compat) and discoverable by tooling.
register_port("SupportsTransactions", SupportsTransactions)
register_port("SupportsSessionFactory", SupportsSessionFactory)

__all__ = ["SupportsSessionFactory", "SupportsTransactions"]
