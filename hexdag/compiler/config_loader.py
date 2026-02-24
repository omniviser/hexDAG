"""Configuration loader for HexDAG.

Parses configuration into kernel domain models. Supports two config sources:

1. **kind: Config YAML** — canonical format, loaded via explicit path or
   ``HEXDAG_CONFIG_PATH`` env var.
2. **pyproject.toml [tool.hexdag]** — auto-discovery fallback (standard Python
   convention).

This is part of the compiler (userspace) — the kernel never touches config
file formats directly.
"""

from __future__ import annotations

import os
import re
import tomllib  # Python 3.11+
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from hexdag.kernel.config.models import (
    DefaultCaps,
    DefaultLimits,
    HexDAGConfig,
    LoggingConfig,
    ManifestEntry,
)
from hexdag.kernel.logging import get_logger
from hexdag.kernel.orchestration.models import OrchestratorConfig

# Type alias for configuration data that can be recursively substituted
ConfigData = str | dict[str, "ConfigData"] | list["ConfigData"] | int | float | bool | None

# Constants for boolean environment variable parsing
_TRUTHY_VALUES = frozenset({"true", "1", "yes", "on", "enabled"})
_FALSY_VALUES = frozenset({"false", "0", "no", "off", "disabled"})

logger = get_logger(__name__)


def _parse_bool_env(value: str) -> bool:
    """Parse boolean from environment variable value.

    Parameters
    ----------
    value : str
        Environment variable value

    Returns
    -------
    bool
        Parsed boolean value

    Raises
    ------
    ValueError
        If value is not a recognized boolean string
    """
    normalized = value.lower().strip()
    if normalized in _TRUTHY_VALUES:
        return True
    if normalized in _FALSY_VALUES:
        return False
    expected = _TRUTHY_VALUES | _FALSY_VALUES
    raise ValueError(f"Invalid boolean value: {value!r}. Expected one of: {expected}")


@lru_cache(maxsize=32)
def _load_and_parse_cached(path_str: str) -> HexDAGConfig:
    """Cached configuration loader."""
    loader = ConfigLoader()
    return loader._load_and_parse(Path(path_str))


class ConfigLoader:
    """Loads and processes HexDAG configuration files.

    Supports two config sources:

    1. ``kind: Config`` YAML manifests (explicit path or env var)
    2. ``pyproject.toml [tool.hexdag]`` (auto-discovery)
    """

    ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")

    def load_config_file(self, path: str | Path | None = None) -> HexDAGConfig:
        """Load configuration from YAML or pyproject.toml.

        Parameters
        ----------
        path : str | Path | None
            Path to config file. If None, searches using discovery order.

        Returns
        -------
        HexDAGConfig
            Parsed configuration with environment variables substituted
        """
        config_path = self._find_config_file(path)
        return _load_and_parse_cached(str(config_path.absolute()))

    def load_from_toml(self, path: str | Path | None = None) -> HexDAGConfig:
        """Load configuration (backward-compatible entry point).

        Parameters
        ----------
        path : str | Path | None
            Path to config file. If None, searches using discovery order.

        Returns
        -------
        HexDAGConfig
            Parsed configuration with environment variables substituted
        """
        return self.load_config_file(path)

    def _load_and_parse(self, config_path: Path) -> HexDAGConfig:
        """Load and parse configuration file (YAML or TOML)."""
        logger.info("Loading configuration from {path}", path=config_path)

        if config_path.suffix in (".yaml", ".yml"):
            return self._load_yaml_config(config_path)
        return self._load_toml_config(config_path)

    def _load_yaml_config(self, config_path: Path) -> HexDAGConfig:
        """Load and parse a kind: Config YAML file.

        Parameters
        ----------
        config_path : Path
            Path to YAML config file

        Returns
        -------
        HexDAGConfig
            Parsed configuration

        Raises
        ------
        ValueError
            If the YAML file is not a valid kind: Config manifest
        """
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(
                f"Invalid YAML config file: expected a mapping, got {type(data).__name__}"
            )

        kind = data.get("kind")
        if kind != "Config":
            raise ValueError(
                f"YAML config file must use 'kind: Config' manifest format. "
                f"Got 'kind: {kind}' in {config_path.name}."
            )

        spec = data.get("spec", {})
        if not isinstance(spec, dict):
            raise ValueError("'spec' field in kind: Config must be a mapping")

        spec = self._substitute_env_vars(spec)
        return self._parse_config(spec)

    def _load_toml_config(self, config_path: Path) -> HexDAGConfig:
        """Load and parse a TOML config file (pyproject.toml).

        Parameters
        ----------
        config_path : Path
            Path to TOML config file

        Returns
        -------
        HexDAGConfig
            Parsed configuration
        """
        with config_path.open("rb") as f:
            data = tomllib.load(f)

        if config_path.name == "pyproject.toml":
            hexdag_data = data.get("tool", {}).get("hexdag", {})
            if not hexdag_data:
                logger.warning("No [tool.hexdag] section found in pyproject.toml, using defaults")
                return get_default_config()
        else:
            # Direct TOML file — check if it has [tool.hexdag] or is flat
            if "tool" in data and "hexdag" in data.get("tool", {}):
                hexdag_data = data["tool"]["hexdag"]
            else:
                hexdag_data = data

        hexdag_data = self._substitute_env_vars(hexdag_data)
        return self._parse_config(hexdag_data)

    def _find_config_file(self, path: str | Path | None) -> Path:
        """Find configuration file.

        Discovery order:
        1. Explicit path argument
        2. ``HEXDAG_CONFIG_PATH`` env var
        3. ``pyproject.toml`` in CWD
        4. ``pyproject.toml`` in parent directories (with ``[tool.hexdag]``)

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

        # Check environment variable
        if env_path := os.getenv("HEXDAG_CONFIG_PATH"):
            config_path = Path(env_path)
            if config_path.exists():
                logger.debug("Using config from HEXDAG_CONFIG_PATH: {}", config_path)
                return config_path
            logger.warning("HEXDAG_CONFIG_PATH set but file not found: {}", config_path)

        # Auto-discovery: pyproject.toml only
        if Path("pyproject.toml").exists():
            return Path("pyproject.toml")

        # Parent directory traversal for pyproject.toml
        current = Path.cwd()
        while current != current.parent:
            pyproject = current / "pyproject.toml"
            if pyproject.exists():
                with pyproject.open("rb") as f:
                    data = tomllib.load(f)
                    if "tool" in data and "hexdag" in data["tool"]:
                        return pyproject
            current = current.parent

        raise FileNotFoundError(
            "No configuration file found. Provide a kind: Config YAML path, "
            "set HEXDAG_CONFIG_PATH, or add [tool.hexdag] to pyproject.toml"
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
                    logger.debug(
                        "Environment variable ${{{var_name}}} not found, keeping placeholder",
                        var_name=var_name,
                    )
                    return match.group(0)  # Keep original placeholder

                return value

            return self.ENV_VAR_PATTERN.sub(replacer, data)

        if isinstance(data, dict):
            return {key: self._substitute_env_vars(value) for key, value in data.items()}

        if isinstance(data, list):
            return [self._substitute_env_vars(item) for item in data]

        return data

    def _parse_config(self, data: dict[str, Any]) -> HexDAGConfig:
        """Parse configuration data into HexDAGConfig.

        Parameters
        ----------
        data : dict[str, Any]
            Raw configuration data (format-agnostic)

        Returns
        -------
        HexDAGConfig
            Parsed configuration object
        """
        config = HexDAGConfig()

        # Parse modules list
        if "modules" in data:
            config.modules = data["modules"]
            logger.debug("Loaded {count} modules", count=len(config.modules))

        # Parse plugins list
        if "plugins" in data:
            config.plugins = data["plugins"]
            logger.debug("Loaded {count} plugins", count=len(config.plugins))

        # Parse dev mode
        config.dev_mode = data.get("dev_mode", False)

        # Parse logging configuration with environment variable overrides
        config.logging = self._parse_logging_config(data.get("logging", {}))

        # Parse settings section
        if "settings" in data:
            config.settings = data["settings"]
            logger.debug("Loaded {count} settings", count=len(config.settings))

        # Parse orchestrator config (supports both "orchestrator" and "kernel" keys)
        orchestrator_data = data.get("orchestrator") or data.get("kernel", {})
        if orchestrator_data:
            config.orchestrator = OrchestratorConfig(
                max_concurrent_nodes=orchestrator_data.get("max_concurrent_nodes", 10),
                strict_validation=orchestrator_data.get("strict_validation", False),
                default_node_timeout=orchestrator_data.get("default_node_timeout"),
            )

        # Parse default resource limits
        if "limits" in data:
            limits_data = data["limits"]
            config.limits = DefaultLimits(
                max_total_tokens=limits_data.get("max_total_tokens"),
                max_llm_calls=limits_data.get("max_llm_calls"),
                max_tool_calls=limits_data.get("max_tool_calls"),
                max_cost_usd=limits_data.get("max_cost_usd"),
                warning_threshold=limits_data.get("warning_threshold", 0.8),
            )

        # Parse default capability boundaries
        if "caps" in data:
            caps_data = data["caps"]
            config.caps = DefaultCaps(
                default_set=caps_data.get("default_set"),
                deny=caps_data.get("deny"),
            )

        return config

    def _parse_logging_config(self, logging_data: dict[str, Any]) -> LoggingConfig:
        """Parse logging configuration with environment variable overrides.

        Environment variables take precedence over config file values:
        - HEXDAG_LOG_LEVEL: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        - HEXDAG_LOG_FORMAT: Output format (console, json, structured, rich, dual)
        - HEXDAG_LOG_FILE: Optional file path for log output
        - HEXDAG_LOG_COLOR: Use color output (true/false)
        - HEXDAG_LOG_TIMESTAMP: Include timestamp (true/false)
        - HEXDAG_LOG_RICH: Use Rich library for enhanced output (true/false)
        - HEXDAG_LOG_DUAL_SINK: Enable dual-sink mode (true/false)
        - HEXDAG_LOG_STDLIB_BRIDGE: Enable stdlib logging bridge (true/false)
        - HEXDAG_LOG_BACKTRACE: Enable backtrace in logs (true/false)
        - HEXDAG_LOG_DIAGNOSE: Enable diagnose mode (true/false)

        Parameters
        ----------
        logging_data : dict[str, Any]
            Logging section from config

        Returns
        -------
        LoggingConfig
            Parsed logging configuration with env overrides applied
        """
        # Start with config values
        level = logging_data.get("level", "INFO")
        format_type = logging_data.get("format", "structured")
        output_file = logging_data.get("output_file")
        use_color = logging_data.get("use_color", True)
        include_timestamp = logging_data.get("include_timestamp", True)
        use_rich = logging_data.get("use_rich", False)
        dual_sink = logging_data.get("dual_sink", False)
        enable_stdlib_bridge = logging_data.get("enable_stdlib_bridge", False)
        backtrace = logging_data.get("backtrace", True)
        diagnose = logging_data.get("diagnose", True)

        # Apply environment variable overrides
        if env_level := os.getenv("HEXDAG_LOG_LEVEL"):
            level = env_level.upper()
            logger.debug("Overriding log level from env: {}", level)

        if env_format := os.getenv("HEXDAG_LOG_FORMAT"):
            format_type = env_format.lower()
            logger.debug("Overriding log format from env: {}", format_type)

        if env_file := os.getenv("HEXDAG_LOG_FILE"):
            output_file = env_file
            logger.debug("Overriding log file from env: {}", output_file)

        if env_color := os.getenv("HEXDAG_LOG_COLOR"):
            try:
                use_color = _parse_bool_env(env_color)
                logger.debug("Overriding log color from env: {}", use_color)
            except ValueError as e:
                logger.warning("Invalid HEXDAG_LOG_COLOR value: {}", e)

        if env_timestamp := os.getenv("HEXDAG_LOG_TIMESTAMP"):
            try:
                include_timestamp = _parse_bool_env(env_timestamp)
                logger.debug("Overriding log timestamp from env: {}", include_timestamp)
            except ValueError as e:
                logger.warning("Invalid HEXDAG_LOG_TIMESTAMP value: {}", e)

        if env_rich := os.getenv("HEXDAG_LOG_RICH"):
            try:
                use_rich = _parse_bool_env(env_rich)
                logger.debug("Overriding use_rich from env: {}", use_rich)
            except ValueError as e:
                logger.warning("Invalid HEXDAG_LOG_RICH value: {}", e)

        if env_dual_sink := os.getenv("HEXDAG_LOG_DUAL_SINK"):
            try:
                dual_sink = _parse_bool_env(env_dual_sink)
                logger.debug("Overriding dual_sink from env: {}", dual_sink)
            except ValueError as e:
                logger.warning("Invalid HEXDAG_LOG_DUAL_SINK value: {}", e)

        if env_stdlib_bridge := os.getenv("HEXDAG_LOG_STDLIB_BRIDGE"):
            try:
                enable_stdlib_bridge = _parse_bool_env(env_stdlib_bridge)
                logger.debug("Overriding enable_stdlib_bridge from env: {}", enable_stdlib_bridge)
            except ValueError as e:
                logger.warning("Invalid HEXDAG_LOG_STDLIB_BRIDGE value: {}", e)

        if env_backtrace := os.getenv("HEXDAG_LOG_BACKTRACE"):
            try:
                backtrace = _parse_bool_env(env_backtrace)
                logger.debug("Overriding backtrace from env: {}", backtrace)
            except ValueError as e:
                logger.warning("Invalid HEXDAG_LOG_BACKTRACE value: {}", e)

        if env_diagnose := os.getenv("HEXDAG_LOG_DIAGNOSE"):
            try:
                diagnose = _parse_bool_env(env_diagnose)
                logger.debug("Overriding diagnose from env: {}", diagnose)
            except ValueError as e:
                logger.warning("Invalid HEXDAG_LOG_DIAGNOSE value: {}", e)

        # Cast to proper Literal types for type safety
        # These will be validated by Pydantic at runtime
        return LoggingConfig(
            level=cast("Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']", level),
            format=cast("Literal['console', 'json', 'structured', 'dual', 'rich']", format_type),
            output_file=output_file,
            use_color=use_color,
            include_timestamp=include_timestamp,
            use_rich=use_rich,
            dual_sink=dual_sink,
            enable_stdlib_bridge=enable_stdlib_bridge,
            backtrace=backtrace,
            diagnose=diagnose,
        )


@lru_cache(maxsize=32)
def _cached_load_config(path_str: str | None) -> HexDAGConfig:
    """Internal cached configuration loader.

    Parameters
    ----------
    path_str : str | None
        String representation of path for caching
    Returns
    -------
    HexDAGConfig
        Loaded configuration
    """
    try:
        loader = ConfigLoader()
        if path_str and path_str.startswith("__auto__"):
            return loader.load_config_file(None)
        return loader.load_config_file(Path(path_str) if path_str else None)
    except FileNotFoundError:
        logger.info("No configuration file found, using defaults")
        return get_default_config()


def load_config(path: str | Path | None = None) -> HexDAGConfig:
    """Load configuration from file or return defaults.

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
        return loader.load_config_file(path)
    except FileNotFoundError:
        logger.info("No configuration file found, using defaults")
        return get_default_config()


def clear_config_cache() -> None:
    """Clear all configuration caches.

    Useful for testing or when configuration files have been modified
    and you need to force a reload.
    """
    _cached_load_config.cache_clear()
    _load_and_parse_cached.cache_clear()


def get_default_config() -> HexDAGConfig:
    """Get default configuration.

    Returns
    -------
    HexDAGConfig
        Default configuration with core ports and builtin components
    """
    return HexDAGConfig(
        modules=[
            "hexdag.kernel.ports",
            "hexdag.stdlib.nodes",
            "hexdag.stdlib.adapters.mock",
            "hexdag.drivers.executors",
            "hexdag.drivers.observer_manager",
            "hexdag.drivers.pipeline_spawner",
            "hexdag.kernel.domain.agent_tools",
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
    # Core and builtin modules go to 'core' namespace, others to 'user'
    module_entries = [
        ManifestEntry(
            namespace=(
                "core"
                if (
                    module.startswith("hexdag.kernel")
                    or module.startswith("hexdag.stdlib")
                    or module.startswith("hexdag.drivers")
                )
                else "user"
            ),
            module=module,
        )
        for module in config.modules
    ]

    # Plugin modules go to 'plugin' namespace
    plugin_entries = [ManifestEntry(namespace="plugin", module=plugin) for plugin in config.plugins]

    return module_entries + plugin_entries
