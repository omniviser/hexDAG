"""Registry for plugin-contributed port protocols.

The kernel defines the *generic* port contracts that the kernel and stdlib
genuinely consume.  Domain-specific contracts — ones only a plugin and its
users care about (database transactions, email, file storage, …) — belong to
the plugin that owns them, alongside their implementations.

This registry is the seam that lets a plugin own its port protocols while
keeping ``hexdag.kernel.ports`` a stable import surface.  A plugin registers
its protocol on import::

    # hexdag_plugins/database/_ports.py
    from hexdag.kernel.ports.registry import register_port

    @runtime_checkable
    class SupportsTransactions(Protocol): ...

    register_port("SupportsTransactions", SupportsTransactions)

and the name keeps resolving for back-compat::

    from hexdag.kernel.ports import SupportsTransactions  # via lazy __getattr__

The kernel never imports the plugin; resolution is lazy and discovery-driven
(see :func:`hexdag.kernel.discovery.load_plugin_ports`).
"""

from __future__ import annotations

from typing import Any

_PORT_REGISTRY: dict[str, Any] = {}


def register_port(name: str, protocol: Any) -> None:
    """Register a plugin-contributed port protocol under ``name``.

    Idempotent — re-registering the same name overwrites the previous entry,
    which keeps repeated plugin imports harmless.

    Parameters
    ----------
    name : str
        The attribute name the protocol should be importable as from
        ``hexdag.kernel.ports`` (e.g. ``"SupportsTransactions"``).
    protocol : Any
        The protocol class (typically a ``runtime_checkable`` ``Protocol``).
    """
    _PORT_REGISTRY[name] = protocol


def get_port(name: str) -> Any | None:
    """Return a registered port protocol, or ``None`` if not registered."""
    return _PORT_REGISTRY.get(name)


def resolve_plugin_port(name: str) -> Any | None:
    """Return a plugin port protocol, triggering plugin discovery on first miss.

    Looks up *name* in the registry; if absent, runs plugin-port discovery
    (which imports plugin ``_ports`` modules that call :func:`register_port`)
    and retries the lookup once.  Returns ``None`` if still unresolved.

    Routing the discovery trigger through here keeps the lazy port resolvers
    (``ports.data_store.__getattr__`` / ``ports.__init__.__getattr__``) from
    importing :mod:`hexdag.kernel.discovery` directly, which would close an
    import cycle (discovery → ports.detection → ports.data_store → discovery).
    """
    proto = _PORT_REGISTRY.get(name)
    if proto is None:
        from hexdag.kernel.discovery import (
            load_plugin_ports,  # lazy: registry→discovery, no path back (acyclic)
        )

        load_plugin_ports()
        proto = _PORT_REGISTRY.get(name)
    return proto


def registered_ports() -> dict[str, Any]:
    """Return a shallow copy of the current port registry."""
    return dict(_PORT_REGISTRY)
