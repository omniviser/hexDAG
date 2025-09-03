"""Type definitions for the registry system."""

from __future__ import annotations

from enum import StrEnum


class ComponentType(StrEnum):
    """Enumeration of component types in the registry."""

    NODE = "node"
    ADAPTER = "adapter"
    TOOL = "tool"
    POLICY = "policy"
    MEMORY = "memory"
    OBSERVER = "observer"
