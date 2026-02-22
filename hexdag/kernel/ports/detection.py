"""Port type detection from adapter classes.

Inspects an adapter's class hierarchy to determine which port protocol
it implements.  Uses ``issubclass`` against actual protocol classes —
no string-based MRO matching.

Marker protocols (empty body, e.g. ``LLM``, ``DataStore``) are matched
by explicit MRO inheritance.  Capability protocols (with ``@abstractmethod``
or ``__protocol_attrs__``, e.g. ``SupportsGeneration``, ``Memory``) are
matched structurally via ``issubclass``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# Port type → protocol classes, checked in order.
# First match wins, so put specific checks before general ones.
_PORT_REGISTRY: Sequence[tuple[str, tuple[type, ...]]] | None = None


def _is_marker_protocol(proto: type) -> bool:
    """Return True if *proto* is a marker protocol with no required members."""
    abstract: frozenset[str] = getattr(proto, "__abstractmethods__", frozenset())
    attrs: set[str] = getattr(proto, "__protocol_attrs__", set())
    return not abstract and not attrs


def _explicit_mro_match(cls: type, proto: type) -> bool:
    """Check if *proto* appears explicitly in *cls*'s MRO (not structural)."""
    return proto in cls.__mro__


def _build_registry() -> Sequence[tuple[str, tuple[type, ...]]]:
    """Lazy-build the port registry to avoid circular imports."""
    from hexdag.kernel.ports.api_call import APICall
    from hexdag.kernel.ports.data_store import DataStore
    from hexdag.kernel.ports.database import Database
    from hexdag.kernel.ports.file_storage import FileStorage
    from hexdag.kernel.ports.llm import (
        LLM,
        SupportsEmbedding,
        SupportsFunctionCalling,
        SupportsGeneration,
        SupportsVision,
    )
    from hexdag.kernel.ports.memory import Memory
    from hexdag.kernel.ports.pipeline_spawner import PipelineSpawner
    from hexdag.kernel.ports.secret import SecretStore
    from hexdag.kernel.ports.tool_router import ToolRouter
    from hexdag.kernel.ports.vector_search import SupportsVectorSearch
    from hexdag.kernel.ports.vfs import VFS

    # Order matters: specific before general.
    return (
        # LLM — check capability sub-protocols first (structural match),
        # then the LLM marker (explicit MRO only).
        (
            "llm",
            (SupportsGeneration, SupportsFunctionCalling, SupportsVision, SupportsEmbedding, LLM),
        ),
        # Memory (deprecated → DataStore, but still detected)
        ("memory", (Memory,)),
        # Database
        ("database", (Database,)),
        # Secret store
        ("secret", (SecretStore,)),
        # File storage
        ("storage", (FileStorage,)),
        # Vector search (standalone, not inside DataStore)
        ("vector_search", (SupportsVectorSearch,)),
        # DataStore — after memory/database so those win when ambiguous
        ("data_store", (DataStore,)),
        # Pipeline spawner
        ("pipeline_spawner", (PipelineSpawner,)),
        # Tool router
        ("tool_router", (ToolRouter,)),
        # VFS
        ("vfs", (VFS,)),
        # API call — last before "unknown" (most generic HTTP port)
        ("api_call", (APICall,)),
    )


def detect_port_type(adapter_class: type) -> str:
    """Detect port type from adapter class using protocol inspection.

    Uses ``issubclass`` for capability protocols (those with required
    methods) and explicit MRO membership for marker protocols (empty
    body).

    Parameters
    ----------
    adapter_class : type
        The adapter class to inspect

    Returns
    -------
    str
        Port type: ``"llm"``, ``"memory"``, ``"database"``, ``"secret"``,
        ``"storage"``, ``"vector_search"``, ``"data_store"``,
        ``"pipeline_spawner"``, ``"tool_router"``, ``"vfs"``,
        ``"api_call"``, or ``"unknown"``

    Examples
    --------
    >>> from hexdag.stdlib.adapters.openai import OpenAIAdapter
    >>> detect_port_type(OpenAIAdapter)
    'llm'
    """
    # Check explicit decorator metadata first (future @adapter decorator)
    explicit_port = getattr(adapter_class, "_hexdag_implements_port", None)
    if explicit_port:
        return str(explicit_port)

    global _PORT_REGISTRY  # noqa: PLW0603
    if _PORT_REGISTRY is None:
        _PORT_REGISTRY = _build_registry()

    for port_type, protocols in _PORT_REGISTRY:
        for proto in protocols:
            try:
                if _is_marker_protocol(proto):
                    # Marker protocol — require explicit inheritance in MRO
                    if _explicit_mro_match(adapter_class, proto):
                        return port_type
                elif issubclass(adapter_class, proto):
                    # Capability protocol — structural subtype check
                    return port_type
            except TypeError:
                # issubclass can raise TypeError for non-class objects
                continue

    return "unknown"
