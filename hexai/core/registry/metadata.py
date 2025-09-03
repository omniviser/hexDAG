"""Component metadata definitions for the registry system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from hexai.core.registry.types import ComponentType


@dataclass
class ComponentMetadata:
    """Metadata for registered components.

    This simplified metadata tracks essential information about components
    while avoiding unnecessary complexity.
    """

    # Required fields
    name: str
    component_type: ComponentType | str
    namespace: str = "core"

    # Optional fields
    description: str = ""
    version: str = "1.0.0"
    author: str = "hexdag"
    is_core: bool = False
    replaceable: bool = False

    # Collections as fields with factory
    tags: frozenset[str] = field(default_factory=frozenset)
    dependencies: frozenset[str] = field(default_factory=frozenset)

    # Runtime fields (not part of __init__)
    component_class: Optional[type] = field(default=None, init=False)
    config_schema: Optional[dict[str, Any]] = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Validate and normalize metadata after initialization."""
        if not isinstance(self.tags, frozenset):
            # Type ignore: mypy infers tags is always frozenset from field
            self.tags = frozenset(self.tags) if self.tags else frozenset()  # type: ignore
        if not isinstance(self.dependencies, frozenset):
            # Type ignore: mypy infers dependencies is always frozenset from field
            self.dependencies = frozenset(self.dependencies) if self.dependencies else frozenset()  # type: ignore
        # Normalize component type
        if isinstance(self.component_type, str):
            self.component_type = ComponentType(self.component_type)
