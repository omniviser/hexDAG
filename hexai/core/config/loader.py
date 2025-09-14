"""TOML configuration loader for HexDAG."""

from __future__ import annotations

import logging
import os
import re
import tomllib  # Python 3.11+
from pathlib import Path
from typing import Any

from hexai.core.config.models import HexDAGConfig, ManifestEntry

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Loads and processes HexDAG configuration from TOML files."""

    ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")

    def __init__(self) -> None:
        """Initialize the config loader."""
        if tomllib is None:
            raise ImportError(
                "TOML support requires 'tomli' for Python < 3.11. Install with: pip install tomli"
            )

    def load_from_toml(self, path: str | Path | None = None) -> HexDAGConfig:
        """Load configuration from TOML file.

        Parameters
        ----------
        path : str | Path | None
            Path to TOML file. If None, searches for pyproject.toml or hexdag.toml

        Returns
        -------
        HexDAGConfig
            Parsed configuration with environment variables substituted

        Raises
        ------
        FileNotFoundError
            If no configuration file is found
        ValueError
            If configuration is invalid
        """
        # Find config file
        config_path = self._find_config_file(path)
        logger.info("Loading configuration from %s", config_path)

        # Load TOML
        with open(config_path, "rb") as f:
            data = tomllib.load(f)

        # Extract hexdag configuration
        if config_path.name == "pyproject.toml":
            # Look for [tool.hexdag] section
            hexdag_data = data.get("tool", {}).get("hexdag", {})
            if not hexdag_data:
                logger.warning("No [tool.hexdag] section found in pyproject.toml, using defaults")
                return get_default_config()
        else:
            # Direct hexdag.toml file
            hexdag_data = data

        # Process environment variable substitution
        hexdag_data = self._substitute_env_vars(hexdag_data)

        # Parse configuration sections
        config = self._parse_config(hexdag_data)

        return config

    def _find_config_file(self, path: str | Path | None) -> Path:
        """Find configuration file.

        Parameters
        ----------
        path : str | Path | None
            Explicit path or None to search

        Returns
        -------
        Path
            Path to configuration file

        Raises
        ------
        FileNotFoundError
            If no configuration file is found
        """
        if path:
            config_path = Path(path)
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            return config_path

        # Search for config files in order of preference
        search_paths = [
            Path("hexdag.toml"),
            Path("pyproject.toml"),
            Path(".hexdag.toml"),
        ]

        for search_path in search_paths:
            if search_path.exists():
                return search_path

        # Also check parent directories for pyproject.toml
        current = Path.cwd()
        while current != current.parent:
            pyproject = current / "pyproject.toml"
            if pyproject.exists():
                with open(pyproject, "rb") as f:
                    data = tomllib.load(f)
                    if "tool" in data and "hexdag" in data["tool"]:
                        return pyproject
            current = current.parent

        raise FileNotFoundError(
            "No configuration file found. Searched for: hexdag.toml, pyproject.toml, .hexdag.toml"
        )

    def _substitute_env_vars(self, data: Any) -> Any:
        """Recursively substitute environment variables in configuration.

        Parameters
        ----------
        data : Any
            Configuration data (dict, list, str, etc.)

        Returns
        -------
        Any
            Data with environment variables substituted
        """
        if isinstance(data, str):
            # Replace ${VAR} with environment variable value
            def replacer(match: re.Match[str]) -> str:
                var_name = match.group(1)
                value = os.environ.get(var_name)
                if value is None:
                    logger.warning(f"Environment variable ${{{var_name}}} not found")
                    return match.group(0)  # Keep original
                return value

            return self.ENV_VAR_PATTERN.sub(replacer, data)

        elif isinstance(data, dict):
            return {key: self._substitute_env_vars(value) for key, value in data.items()}

        elif isinstance(data, list):
            return [self._substitute_env_vars(item) for item in data]

        else:
            return data

    def _parse_config(self, data: dict[str, Any]) -> HexDAGConfig:
        """Parse configuration data into HexDAGConfig.

        Parameters
        ----------
        data : dict[str, Any]
            Raw configuration data from TOML

        Returns
        -------
        HexDAGConfig
            Parsed configuration object
        """
        config = HexDAGConfig()

        # Parse modules list
        if "modules" in data:
            config.modules = data["modules"]
            logger.debug("Loaded %d modules", len(config.modules))

        # Parse plugins list
        if "plugins" in data:
            config.plugins = data["plugins"]
            logger.debug("Loaded %d plugins", len(config.plugins))

        # Parse dev mode
        config.dev_mode = data.get("dev_mode", False)

        # Parse settings section
        if "settings" in data:
            config.settings = data["settings"]
            logger.debug("Loaded %d settings", len(config.settings))

        return config


def load_config(path: str | Path | None = None) -> HexDAGConfig:
    """Load configuration from TOML file or return defaults.

    Parameters
    ----------
    path : str | Path | None
        Path to configuration file or None to search

    Returns
    -------
    HexDAGConfig
        Loaded configuration or defaults if no file found
    """
    try:
        loader = ConfigLoader()
        return loader.load_from_toml(path)
    except FileNotFoundError:
        logger.info("No configuration file found, using defaults")
        return get_default_config()


def get_default_config() -> HexDAGConfig:
    """Get default configuration.

    Returns
    -------
    HexDAGConfig
        Default configuration with minimal modules
    """
    return HexDAGConfig(
        modules=[
            "hexai.core.application.nodes",
            "hexai.adapters.mock",
        ],
        settings={
            "log_level": "INFO",
            "enable_metrics": True,
        },
    )


def config_to_manifest_entries(config: HexDAGConfig) -> list[ManifestEntry]:
    """Convert configuration to manifest entries.

    Parameters
    ----------
    config : HexDAGConfig
        Configuration object

    Returns
    -------
    list[ManifestEntry]
        List of manifest entries for registry bootstrap
    """
    # Core modules go to 'core' namespace, others to 'user'
    module_entries = [
        ManifestEntry(
            namespace="core" if module.startswith("hexai.core") else "user", module=module
        )
        for module in config.modules
    ]

    # Plugin modules go to 'plugin' namespace
    plugin_entries = [ManifestEntry(namespace="plugin", module=plugin) for plugin in config.plugins]

    return module_entries + plugin_entries
