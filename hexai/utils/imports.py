"""Optional import. Import an optional dependency."""

from __future__ import annotations

import importlib
from typing import Any

from .optional_deps import get_pkg_feature

PYPI_TO_MODULE: dict[str, str] = {
    "pyyaml": "yaml",
    "pydantic-core": "pydantic_core",
}


def _import_name(pypi_pkg: str) -> str:
    """If user calls pyyaml there is no error in name."""
    return PYPI_TO_MODULE.get(pypi_pkg, pypi_pkg)


def optional_import(pkg: str, feature: str | None = None) -> Any:
    """Import an optional dependency at runtime with a clear install hint."""
    features = get_pkg_feature()
    suggested_feature = feature or features.get(pkg)

    try:
        return importlib.import_module(_import_name(pkg))
    except ImportError as e:
        hint = (
            f" Install with: pip install hexdag[{suggested_feature}]" if suggested_feature else ""
        )
        raise RuntimeError(f"Missing optional dependency '{pkg}'.{hint}") from e
