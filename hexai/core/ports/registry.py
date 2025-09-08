"""Runtime Port Registry for dynamic registration, lookup, and instantiation of ports."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Literal, Type

# Allowed kinds of ports
# ruff: formatter-ignore
PortKind = Literal["llm", "database", "memory", "embedding_selector", "ontology", "tool_router"]


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
    health : str | None
        Optional hint about health status of the port.
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
    health: str | None = None
    version: str | None = None


class PortRegistry:
    """In-memory registry of port classes with discoverability and simple instantiation."""

    _registry: ClassVar[dict[str, PortInfo]] = {}

    @classmethod
    def register(cls, name: str, port_cls: Type[Any], **meta: Any) -> None:
        """Register a port implementation under a unique name.

        Parameters
        ----------
        name : str
            Unique identifier for the port.
        port_cls : type[Any]
            Implementation class for the port.
        """
        if name in cls._registry:
            existing = cls._registry[name]
            same_cls = existing.port_cls is port_cls
            same_meta = all(getattr(existing, atr) == i for atr, i in meta.items())
            if same_cls or same_meta:
                return
            raise ValueError(
                f"Port '{name}' already registered with a different definition. "
                f"Use PortRegistry.override(...) if you want to replace it."
            )
        cls._registry[name] = PortInfo(name=name, port_cls=port_cls, **meta)
        logging.info(f"Port '{name}' registered.")

    @classmethod
    def override(cls, name: str, port_cls: Type[Any], **meta: Any) -> None:
        """Force override of an existing definition."""
        if name in cls._registry:
            logging.warning(f"Port '{name}' is being overridden in the registry.")
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
        # copy dict to not expose mutable interior
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


# flake8: noqa
def register_port[T: type](name: str, override: bool = False, **meta: Any) -> Callable[[T], T]:
    """Decorator to register or override a port class in the PortRegistry.

    Args
    ----
        name (str): The name of the port to register.
        override (bool, optional): Whether to override an existing port. Defaults to False.
        **meta (Any): Additional metadata for the port.
    Returns
    -------
        Callable[[T], T]: A decorator for the port class.
    """

    def decorator(cls: T) -> T:
        """Register or override a port class in the PortRegistry."""
        if name in PortRegistry._registry and not override:
            raise ValueError(f"Port '{name}' already exists. Use override=True to replace it.")
        if override:
            PortRegistry.override(name=name, port_cls=cls, **meta)
        else:
            PortRegistry.register(name=name, port_cls=cls, **meta)
        return cls

    return decorator
