"""Import compatibility layer for optional features with helpful error messages.

This module provides a safe way to use optional dependencies in hexDAG. It allows lazy imports
and raises user-friendly errors if a required feature or package is missing, including system
dependencies like Graphviz's `dot`.
"""

from __future__ import annotations

import importlib.util
import shutil
from typing import Iterable

PKG_TO_FEATURE: dict[str, str] = {
    "yaml": "cli",
    "click": "cli",
    "rich": "cli",
    "graphviz": "viz",
    "openai": "adapters-openai",
    "anthropic": "adapters-anthropic",
}


class FeatureMissingError(ImportError):
    """Exception raised when a user tries to use a feature without installing required extras.

    Attributes
    ----------
    message : str
        Human-readable error message explaining which packages are missing.
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class FeatureManager:
    """Manager for optional features and their required packages.

    This class maps feature names to Python packages and checks if the features are available.
    It can also raise informative errors suggesting how to install missing packages.
    """

    _FEATURES: dict[str, list[str]] = {
        feature: [pkg for pkg, f in PKG_TO_FEATURE.items() if f == feature]
        for feature in set(PKG_TO_FEATURE.values())
    }

    @staticmethod
    def _has_package(pkg: str) -> bool:
        """Check if a Python package is installed.

        Parameters
        ----------
        pkg : str
            Name of the Python package to check.

        Returns
        -------
        bool
            True if package is installed, False otherwise.
        """
        return importlib.util.find_spec(pkg) is not None

    @classmethod
    def has_feature(cls, name: str) -> bool:
        """Check if a feature is available.

        This verifies all required Python packages and system dependencies.

        Parameters
        ----------
        name : str
            Name of the feature.

        Returns
        -------
        bool
            True if all required packages (and system deps) are available, False otherwise.
        """
        pkgs = cls._FEATURES.get(name, [])
        if not pkgs:
            return False

        ok = all(cls._has_package(p) for p in pkgs)

        if name == "viz":
            ok = ok and shutil.which("dot") is not None

        return ok

    @classmethod
    def require_feature(cls, name: str) -> None:
        """Raise a friendly error if a feature is not available.

        Parameters
        ----------
        name : str
            Name of the feature.

        Raises
        ------
        FeatureMissingError
            If the feature is missing, with instructions on how to install it.
        """
        if cls.has_feature(name):
            return

        pkgs = cls._FEATURES.get(name, [])
        missing = [p for p in pkgs if not cls._has_package(p)]
        msg = [f"Feature '{name}' is not available."]

        if missing:
            msg.append(f"Missing Python packages: {', '.join(missing)}.")

        msg.append(f"Install with: pip install hexdag[{name}]")

        if name == "viz" and shutil.which("dot") is None:
            msg.append(
                "Also missing system Graphviz ('dot'). "
                "On Ubuntu/WSL: sudo apt-get update && sudo apt-get install graphviz"
            )

        raise FeatureMissingError("\n".join(msg))

    @classmethod
    def require_packages(cls, pkgs: Iterable[str], feature_hint: str | None = None) -> None:
        """Require specific packages with optional feature hint.

        Parameters
        ----------
        pkgs : Iterable[str]
            List of Python packages to check.
        feature_hint : str | None
            Name of the feature for installation suggestion.

        Raises
        ------
        FeatureMissingError
            If any of the packages are missing.
        """
        missing = [p for p in pkgs if not cls._has_package(p)]
        if not missing:
            return
        hint = f" Install with: pip install hexdag[{feature_hint}]" if feature_hint else ""
        raise FeatureMissingError(f"Missing packages: {', '.join(missing)}.{hint}")
