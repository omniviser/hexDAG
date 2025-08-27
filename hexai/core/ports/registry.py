"""Runtime Port Registry for dynamic registration, lookup, and instantiation of ports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Literal, Type

# Allowed kinds of ports
PortKind = Literal[
    "llm", "database", "memory", "embedding_selector", "ontology", "tool_router"
]  # ruff: formatter-ignore

# Health states for a port
HealthStatus = Literal["unknown", "healthy", "unhealthy"]


@dataclass
class PortInfo:
    """Metadata describing a port in the registry.

    Attributes
    ----------
    name : str
        Unique name of the port.
    port_cls : type[Any]
        The class implementing the port.
    kind : PortKind
        Type of the port (llm, database, memory, embedding_selector, ontology, tool_router).
    sync : bool
        Whether the port operates synchronously (default True).
    streaming : bool
        Whether the port supports streaming responses (default False).
    transactional : bool
        Whether the port supports transactions (default False).
    idempotent : bool
        Whether repeated calls produce the same result (default True).
    cost_hint : str | None
        Optional hint about cost of using the port.
    latency_hint : str | None
        Optional hint about expected latency.
    health : HealthStatus
        Current health status of the port (default "unknown").
    version : str | None
        Optional version of the port implementation.
    """

    name: str
    port_cls: Type[Any]
    kind: PortKind
    sync: bool = True
    streaming: bool = False
    transactional: bool = False
    idempotent: bool = True
    cost_hint: str | None = None
    latency_hint: str | None = None
    health: HealthStatus = "unknown"
    version: str | None = None


class PortRegistry:
    """In-memory registry of port classes with discoverability and simple instantiation."""

    _registry: ClassVar[dict[str, PortInfo]] = {}

    @classmethod
    def register(cls, name: str, port_cls: Type[Any], **meta: Any) -> None:
        """Register a port implementation under a unique name."""
        if name in cls._registry:
            raise ValueError(f"Port {name} already registered")
        cls._registry[name] = PortInfo(name=name, port_cls=port_cls, **meta)

    @classmethod
    def unregister(cls, name: str) -> None:
        """Unregister a port implementation under a unique name."""
        if name not in cls._registry:
            raise KeyError(f"Port {name} not registered")
        del cls._registry[name]

    @classmethod
    def get(cls, name: str) -> Type[Any]:
        """Retrieve a port implementation under a unique name."""
        try:
            return cls._registry[name].port_cls
        except KeyError as e:
            raise KeyError(
                f"Port {name} not found. Available ports are {list(cls._registry.keys())}"
            ) from e

    @classmethod
    def meta(cls, name: str) -> dict[str, Any]:
        """Retrieve a port metadata under a unique name."""
        info = cls._registry.get(name)
        if info is None:
            raise KeyError(
                f"Port {name} not registered. Available ports are {list(cls._registry.keys())}"
            )
        # kopia dict, by nie wystawiać mutowalnego wnętrza
        return dict(info.__dict__)

    @classmethod
    def find(cls, **filters: Any) -> dict[str, PortInfo]:
        """Find ports matching given metadata filters.

        Parameters
        ----------
        **filters : Any
            Arbitrary metadata attributes of :class:`PortInfo` to filter on.
            Each keyword argument corresponds to a field in ``PortInfo``
            (e.g., ``kind``, ``streaming``, ``sync``). Only ports where
            all provided attributes match the given values will be returned.

        Returns
        -------
        Dict[str, PortInfo]
            A mapping of port names to their corresponding metadata objects.

        Example
        -------
            PortRegistry.find(kind="llm", streaming=True)
        """
        results: dict[str, PortInfo] = {}
        for name, meta in cls._registry.items():
            if all(getattr(meta, key, None) == value for key, value in filters.items()):
                results[name] = meta
        return results

    @classmethod
    def all(cls) -> dict[str, Type[Any]]:
        """Return all registered ports."""
        return {name: meta.port_cls for name, meta in cls._registry.items()}


def register_port[T: type[Any]](name: str, **meta: Any) -> Callable[[T], T]:
    """Decorator to register a custom port in the PortRegistry.

    Automatically registers the decorated class as a port under a unique name
    with optional metadata (kind, streaming, sync, etc.).
    """

    def decorator(cls: T) -> T:
        PortRegistry.register(name=name, port_cls=cls, **meta)
        return cls

    return decorator
