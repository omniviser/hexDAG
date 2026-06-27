"""hexDAG database plugin — declarative transactions + ready-made DB service.

Owns the transaction machinery that used to live in hexDAG core:

- ``RunScopedResource`` / ``DualModeSessionMixin`` — the dual-mode session
  primitive (standalone autocommit vs. per-run atomic transaction).
- ``SupportsTransactions`` / ``SupportsSessionFactory`` — the transaction port
  protocols (also registered back into ``hexdag.kernel.ports`` for back-compat).
- ``SQLAlchemyAdapter`` / ``SQLAlchemyStateBackend`` — SQLAlchemy implementations.
- ``DatabaseService`` — a ``@tool``/``@step`` CRUD + query service.
- ``TransactionMacro`` (``macro: transaction``) — savepoint-scoped partial rollback.

Importing this package is cheap: only ``_ports`` (which registers the port
protocols) loads eagerly; the SQLAlchemy-backed classes import on first access.
"""

from typing import Any

# Eagerly register the port protocols (light — no SQLAlchemy import).
from hexdag_plugins.database import _ports as _ports  # noqa: F401

_LAZY: dict[str, tuple[str, str]] = {
    "RunScopedResource": ("hexdag_plugins.database.run_scope", "RunScopedResource"),
    "DualModeSessionMixin": ("hexdag_plugins.database.dual_mode", "DualModeSessionMixin"),
    "SQLAlchemyAdapter": ("hexdag_plugins.database.adapters", "SQLAlchemyAdapter"),
    "SQLAlchemyStateBackend": ("hexdag_plugins.database.adapters", "SQLAlchemyStateBackend"),
    "DatabaseService": ("hexdag_plugins.database.service", "DatabaseService"),
    "TransactionMacro": ("hexdag_plugins.database.macros.transaction", "TransactionMacro"),
    "SupportsTransactions": ("hexdag_plugins.database._ports", "SupportsTransactions"),
    "SupportsSessionFactory": ("hexdag_plugins.database._ports", "SupportsSessionFactory"),
}


def __getattr__(name: str) -> Any:  # PEP 562 — lazy, avoids importing SQLAlchemy at package load
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(target[0])
    return getattr(module, target[1])


def __dir__() -> list[str]:
    return sorted([*globals().keys(), *_LAZY.keys()])


__all__ = list(_LAZY.keys())
