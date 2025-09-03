"""Registry system for dynamic component discovery and management."""

from hexai.core.registry.decorators import (
    adapter,
    component,
    get_component_metadata,
    memory,
    node,
    observer,
    policy,
    tool,
)
from hexai.core.registry.discovery import discover_entry_points
from hexai.core.registry.metadata import ComponentMetadata
from hexai.core.registry.registry import ComponentRegistry, registry
from hexai.core.registry.types import ComponentType

__all__ = [
    # Singleton registry (primary interface)
    "registry",
    "ComponentRegistry",
    # Component metadata
    "ComponentMetadata",
    "ComponentType",
    # Decorators for components (auto-register)
    "component",
    "node",
    "tool",
    "adapter",
    "policy",
    "memory",
    "observer",
    "get_component_metadata",
    # Plugin support
    "discover_entry_points",
]
