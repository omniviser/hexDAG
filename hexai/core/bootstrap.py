"""Bootstrap the HexDAG registry from TOML configuration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from hexai.core.config import config_to_manifest_entries, load_config
from hexai.core.registry import registry

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def bootstrap_registry(
    config_path: str | Path | None = None,
    dev_mode: bool | None = None,
) -> None:
    """Bootstrap the component registry from TOML configuration.

    This is the standard way to initialize HexDAG at application startup.
    It loads configuration from TOML and populates the registry.

    Parameters
    ----------
    config_path : str | Path | None
        Path to TOML configuration (pyproject.toml or hexdag.toml).
        If None, searches for configuration files automatically.
    dev_mode : bool | None
        Whether to enable development mode (allows post-bootstrap registration).
        If None, uses the value from configuration.

    Examples
    --------
    >>> # Use auto-discovered TOML config
    >>> bootstrap_registry()

    >>> # Use specific TOML config
    >>> bootstrap_registry("custom.toml")

    >>> # Force dev mode
    >>> bootstrap_registry(dev_mode=True)


    """
    # Check if already bootstrapped
    if registry.ready:
        logger.warning("Registry already bootstrapped, skipping")
        return

    # Load configuration
    config = load_config(config_path)

    # Clear precedence: parameter > config > default(False)
    final_dev_mode = dev_mode if dev_mode is not None else config.dev_mode

    # Convert config to manifest entries and bootstrap
    entries = config_to_manifest_entries(config)
    registry.bootstrap(entries, dev_mode=final_dev_mode)

    logger.info("Registry initialized with %d components", len(registry.list_components()))


def ensure_bootstrapped(config_path: str | Path | None = None) -> None:
    """Ensure the registry is bootstrapped, initializing if needed.

    This is a convenience function that can be called multiple times safely.
    It only bootstraps if the registry isn't already ready.

    Parameters
    ----------
    config_path : str | Path | None
        Path to TOML configuration file. If None, auto-discovers config.
    """
    if not registry.ready:
        bootstrap_registry(config_path)
