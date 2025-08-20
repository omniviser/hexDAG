"""Optional import. Import an optional dependency."""

from __future__ import annotations

import importlib
from typing import Any

PKG_TO_FEATURE = {
    "yaml": "cli",
    "click": "cli",
    "rich": "cli",
    "graphviz": "viz",
    "openai": "adapters-openai",
    "anthropic": "adapters-anthropic",
}


def optional_import(pkg: str, feature: str | None = None) -> Any:
    """Import an optional dependency at runtime with a clear install hint."""
    try:
        return importlib.import_module(pkg)
    except ImportError as e:
        feat = feature or PKG_TO_FEATURE.get(pkg)
        hint = f" Install with: pip install hexdag[{feat}]" if feat else ""
        raise RuntimeError(f"Missing optional dependency '{pkg}'.{hint}") from e
