"""Runtime Port Registry for dynamic registration, lookup, and instantiation of ports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Dict, Literal, Type, TypeVar

PortKind = Literal["llm", "database", "memory", "embedding_selector", "ontology", "tool_router"]


@dataclass
class PortInfo:
    """Metadata describing a port."""

    name: str
    port_cls: Type[Any]
    kind: PortKind
    sync: bool = True
    streaming: bool = False
    transactional: bool = False
    idempotent: bool = True
    cost_hint: str | None = None
    latency_hint: str | None = None
    health: str = "unknown"
    version: str | None = None


class PortRegistry:
    """In-memory registry of port classes with discoverability and simple instantiation."""

    _registry: ClassVar[Dict[str, PortInfo]] = {}

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
    def meta(cls, name: str) -> Dict[str, Any]:
        """Retrieve a port metadata under a unique name."""
        info = cls._registry.get(name)
        if info is None:
            raise KeyError(
                f"Port {name} not registered. Available ports are {list(cls._registry.keys())}"
            )
        # kopia dict, by nie wystawiać mutowalnego wnętrza
        return dict(info.__dict__)

    @classmethod
    def find(cls, **filters: Any) -> Dict[str, PortInfo]:
        """
        Find ports matching given metadata filters.

        Example
        -------
            PortRegistry.find(kind="llm", streaming=True)
        """
        results: Dict[str, PortInfo] = {}
        for name, meta in cls._registry.items():
            if all(getattr(meta, key, None) == value for key, value in filters.items()):
                results[name] = meta
        return results

    @classmethod
    def all(cls) -> Dict[str, Type[Any]]:
        """Return all registered ports."""
        return {name: meta.port_cls for name, meta in cls._registry.items()}


T = TypeVar("T", bound=Type[Any])


def register_port(name: str, **meta: Any) -> Callable[[T], T]:
    """
    Decorator to register a custom port in the PortRegistry.

    Automatically registers the decorated class as a port under a unique name
    with optional metadata (kind, streaming, sync, etc.).
    """

    def decorator(cls: T) -> T:
        PortRegistry.register(name=name, port_cls=cls, **meta)
        return cls

    return decorator
