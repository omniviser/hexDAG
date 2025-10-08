"""Example: Configuration-Based Logging

Demonstrates how to configure logging via hexDAG configuration files (TOML/YAML).

This is the recommended approach for production applications as it:
- Centralizes all configuration in one place
- Supports environment variable overrides
- Integrates with hexDAG's bootstrap process
- Makes logging configuration declarative and version-controlled
"""

import os
import tempfile
from pathlib import Path

from hexdag.core.bootstrap import bootstrap_registry
from hexdag.core.config import load_config
from hexdag.core.logging import get_logger


def example_toml_logging():
    """Configure logging via TOML file."""
    print("\n=== Logging from TOML Configuration ===\n")

    # Create a temporary TOML config
    config_content = """
[tool.hexdag]
dev_mode = true

[tool.hexdag.logging]
level = "DEBUG"
format = "structured"
use_color = true
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(config_content)
        config_path = f.name

    try:
        # Bootstrap hexDAG with config - this automatically configures logging!
        bootstrap_registry(config_path)

        # Get a logger and use it
        logger = get_logger(__name__)
        logger.debug("Debug message from TOML config")
        logger.info("Info message from TOML config")
        logger.warning("Warning message from TOML config")

    finally:
        # Cleanup
        Path(config_path).unlink()


def example_env_override():
    """Show environment variable override of TOML config."""
    print("\n\n=== Environment Variable Override ===\n")

    # Create a TOML config with INFO level
    config_content = """
[tool.hexdag.logging]
level = "INFO"
format = "structured"
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(config_content)
        config_path = f.name

    try:
        # Set environment variable to override level to DEBUG
        os.environ["HEXDAG_LOG_LEVEL"] = "DEBUG"
        os.environ["HEXDAG_LOG_FORMAT"] = "json"

        # Load config - env vars take precedence
        config = load_config(config_path)

        print("Config level (from TOML): INFO")
        print(f"Actual level (from env): {config.logging.level}")
        print("Config format (from TOML): structured")
        print(f"Actual format (from env): {config.logging.format}")

    finally:
        # Cleanup
        Path(config_path).unlink()
        del os.environ["HEXDAG_LOG_LEVEL"]
        del os.environ["HEXDAG_LOG_FORMAT"]


def example_production_config():
    """Production logging configuration."""
    print("\n\n=== Production Configuration ===\n")

    config_content = """
[tool.hexdag]
dev_mode = false

[tool.hexdag.logging]
level = "INFO"
format = "json"
output_file = "/tmp/hexdag_production.log"
use_color = false
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(config_content)
        config_path = f.name

    try:
        config = load_config(config_path)

        print("Production logging config:")
        print(f"  Level: {config.logging.level}")
        print(f"  Format: {config.logging.format}")
        print(f"  Output File: {config.logging.output_file}")
        print(f"  Use Color: {config.logging.use_color}")

    finally:
        Path(config_path).unlink()


def example_existing_config():
    """Load from existing config file."""
    print("\n\n=== Using Existing Config File ===\n")

    # The examples/configs directory has a sample config
    config_file = Path(__file__).parent / "configs" / "hexdag_with_logging.toml"

    if config_file.exists():
        print(f"Loading from: {config_file}")
        config = load_config(config_file)

        print("\nLogging Configuration:")
        print(f"  Level: {config.logging.level}")
        print(f"  Format: {config.logging.format}")
        print(f"  Use Color: {config.logging.use_color}")
        print(f"  Include Timestamp: {config.logging.include_timestamp}")
    else:
        print(f"Config file not found: {config_file}")


def main():
    """Run all configuration-based logging examples."""
    print("=" * 60)
    print("hexDAG Configuration-Based Logging Examples")
    print("=" * 60)

    example_toml_logging()
    example_env_override()
    example_production_config()
    example_existing_config()

    print("\n\n" + "=" * 60)
    print("Configuration Guide:")
    print("=" * 60)
    print(
        """
## TOML Configuration (pyproject.toml or hexdag.toml):

[tool.hexdag.logging]
level = "DEBUG"           # DEBUG, INFO, WARNING, ERROR, CRITICAL
format = "structured"     # console, json, structured
output_file = "app.log"   # Optional file output
use_color = true          # ANSI color codes
include_timestamp = true  # Include timestamps

## Environment Variable Overrides:

export HEXDAG_LOG_LEVEL=DEBUG
export HEXDAG_LOG_FORMAT=json
export HEXDAG_LOG_FILE=/var/log/hexdag/app.log
export HEXDAG_LOG_COLOR=true
export HEXDAG_LOG_TIMESTAMP=true

## Usage in Code:

from hexdag.core.bootstrap import bootstrap_registry
from hexdag.core.logging import get_logger

# Bootstrap automatically configures logging from config file
bootstrap_registry()

# Use logger normally
logger = get_logger(__name__)
logger.info("Application started")

## Priority Order (highest to lowest):

1. Environment variables (HEXDAG_LOG_*)
2. TOML configuration ([tool.hexdag.logging])
3. Defaults (INFO level, structured format)
"""
    )


if __name__ == "__main__":
    main()
