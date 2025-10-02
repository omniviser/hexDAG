"""Configuration data models for HexDAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hexai.core.exceptions import ValidationError


@dataclass
class ManifestEntry:
    """Single entry defining a module to load."""

    namespace: str
    module: str

    def __post_init__(self) -> None:
        """Validate manifest entry.

        Raises
        ------
        ValidationError
            If namespace or module is empty or namespace contains ':'
        """
        if not self.namespace:
            raise ValidationError("namespace", "cannot be empty")
        if not self.module:
            raise ValidationError("module", "cannot be empty")
        if ":" in self.namespace:
            raise ValidationError("namespace", "cannot contain ':'", self.namespace)


@dataclass
class HexDAGConfig:
    """Complete HexDAG configuration."""

    # Core configuration
    modules: list[str] = field(default_factory=list)
    plugins: list[str] = field(default_factory=list)

    # Development settings
    dev_mode: bool = False

    # Additional settings
    settings: dict[str, Any] = field(default_factory=dict)
