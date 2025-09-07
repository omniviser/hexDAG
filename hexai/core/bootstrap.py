"""Bootstrap utilities for initializing the HexDAG registry.

This module provides the standard way to bootstrap the component registry
from the YAML manifest. It's the entry point for application initialization.
"""

from __future__ import annotations

import logging
from pathlib import Path

from hexai.core.registry import registry
from hexai.core.registry.manifest import get_default_manifest, load_manifest_from_yaml

logger = logging.getLogger(__name__)


def bootstrap_registry(
    manifest_path: str | Path | None = None,
    dev_mode: bool | None = None,
) -> None:
    """Bootstrap the component registry from a manifest.

    This is the standard way to initialize HexDAG at application startup.
    It loads the component manifest and populates the registry.

    Parameters
    ----------
    manifest_path : str | Path | None
        Path to a custom manifest YAML file. If None, uses the default
        manifest at hexai/core/component_manifest.yaml.
    dev_mode : bool | None
        Whether to enable development mode (allows post-bootstrap registration).
        If None, uses the value from the manifest config or defaults to False.

    Examples
    --------
    >>> # Use default manifest
    >>> bootstrap_registry()

    >>> # Use custom manifest
    >>> bootstrap_registry("my_project/manifest.yaml")

    >>> # Force dev mode
    >>> bootstrap_registry(dev_mode=True)

    Raises
    ------
    RegistryAlreadyBootstrappedError
        If the registry has already been bootstrapped.
    FileNotFoundError
        If a custom manifest_path is provided but doesn't exist.
    """
    # Check if already bootstrapped
    if registry.ready:
        logger.warning("Registry already bootstrapped, skipping")
        return

    # Load manifest
    if manifest_path:
        logger.info(f"Loading manifest from {manifest_path}")
        manifest = load_manifest_from_yaml(manifest_path)
    else:
        logger.info("Loading default HexDAG manifest")
        manifest = get_default_manifest()

    # Determine dev_mode from environment if not specified
    if dev_mode is None:
        import os

        dev_mode = os.getenv("HEXDAG_DEV_MODE", "false").lower() == "true"

    # Bootstrap the registry
    logger.info(f"Bootstrapping registry (dev_mode={dev_mode})")
    registry.bootstrap(manifest, dev_mode=dev_mode)

    logger.info(f"Registry initialized with {len(registry.list_components())} components")


def ensure_bootstrapped(manifest_path: str | Path | None = None) -> None:
    """Ensure the registry is bootstrapped, initializing if needed.

    This is a convenience function that can be called multiple times safely.
    It only bootstraps if the registry isn't already ready.

    Parameters
    ----------
    manifest_path : str | Path | None
        Path to a custom manifest YAML file. If None, uses the default.

    Examples
    --------
    >>> # Safe to call multiple times
    >>> ensure_bootstrapped()
    >>> ensure_bootstrapped()  # Does nothing if already bootstrapped
    """
    if not registry.ready:
        bootstrap_registry(manifest_path)


# Convenience function for common use case
def init_hexdag(dev_mode: bool = False) -> None:
    """Initialize HexDAG with default settings.

    This is the simplest way to get started with HexDAG.

    Parameters
    ----------
    dev_mode : bool
        Whether to enable development mode.

    Examples
    --------
    >>> # Production initialization
    >>> init_hexdag()

    >>> # Development initialization
    >>> init_hexdag(dev_mode=True)
    """
    bootstrap_registry(dev_mode=dev_mode)
