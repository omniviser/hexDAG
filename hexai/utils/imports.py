"""Optional import. Import an optional dependency."""

from __future__ import annotations

import importlib
from typing import Any

from .optional_deps import FEATURES


def optional_import(pkg: str, feature: str | None = None) -> Any:
    """Import an optional dependency at runtime with a clear install hint."""
    feature = None

    for feat, deps in FEATURES.items():
        if any(pkg in dep for dep in deps):
            feature = feat
            break
    try:
        return importlib.import_module(pkg)
    except ImportError as e:
        hint = f" Install with: pip install hexdag[{feature}]" if feature else ""
        raise RuntimeError(f"Missing optional dependency '{pkg}'.{hint}") from e
