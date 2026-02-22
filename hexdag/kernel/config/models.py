"""Configuration data models for HexDAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from hexdag.kernel.exceptions import ValidationError


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """Logging configuration for HexDAG.

    Attributes
    ----------
    level : str, default="INFO"
        Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format : str, default="structured"
        Output format (console, json, structured, dual, rich)
    output_file : str | None, default=None
        Optional file path to write logs to
    use_color : bool, default=True
        Use ANSI color codes (auto-disabled for non-TTY)
    include_timestamp : bool, default=True
        Include timestamp in log output
    use_rich : bool, default=False
        Use Rich library for enhanced console output with better formatting
    dual_sink : bool, default=False
        Enable dual-sink mode: pretty console (Rich) + structured JSON to stdout
    enable_stdlib_bridge : bool, default=False
        Enable interception of stdlib logging for third-party libraries
    backtrace : bool, default=True
        Enable backtrace for debugging (disable in production for security)
    diagnose : bool, default=True
        Enable diagnose mode with variable values (disable in production for security)

    Examples
    --------
    TOML configuration:

    ```toml
    [tool.hexdag.logging]
    level = "DEBUG"
    format = "rich"
    use_rich = true
    dual_sink = false
    ```

    Dual-sink configuration (dev mode):

    ```toml
    [tool.hexdag.logging]
    level = "INFO"
    dual_sink = true
    use_rich = true
    ```

    Environment variable overrides:

    ```bash
    export HEXDAG_LOG_LEVEL=DEBUG
    export HEXDAG_LOG_FORMAT=rich
    export HEXDAG_LOG_FILE=/var/log/hexdag/app.log
    export HEXDAG_LOG_DUAL_SINK=true
    export HEXDAG_LOG_RICH=true
    ```
    """

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["console", "json", "structured", "dual", "rich"] = "structured"
    output_file: str | None = None
    use_color: bool = True
    include_timestamp: bool = True
    use_rich: bool = False
    dual_sink: bool = False
    enable_stdlib_bridge: bool = False
    backtrace: bool = True
    diagnose: bool = True


@dataclass(frozen=True, slots=True)
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


@dataclass(slots=True)
class HexDAGConfig:
    """Complete HexDAG configuration.

    Attributes
    ----------
    modules : list[str]
        List of module paths to load
    plugins : list[str]
        List of plugin names to load
    dev_mode : bool
        Enable development mode features
    logging : LoggingConfig
        Logging configuration
    settings : dict[str, Any]
        Additional custom settings

    Examples
    --------
    TOML configuration in pyproject.toml:

    ```toml
    [tool.hexdag]
    modules = ["myapp.adapters", "myapp.nodes"]
    plugins = ["hexdag-openai", "hexdag-postgres"]
    dev_mode = true

    [tool.hexdag.logging]
    level = "DEBUG"
    format = "structured"
    use_color = true
    ```
    """

    # Core configuration
    modules: list[str] = field(default_factory=list)
    plugins: list[str] = field(default_factory=list)

    # Development settings
    dev_mode: bool = False

    # Logging configuration
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Additional settings
    settings: dict[str, Any] = field(default_factory=dict)
