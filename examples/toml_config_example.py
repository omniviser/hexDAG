"""Example demonstrating TOML configuration for HexDAG.

Shows how to use TOML files to configure:
- Which modules to load
- Development mode settings
- Default bindings (stored for future use)
- Environment-specific configurations
"""

import tempfile
from pathlib import Path

from hexai.core.bootstrap import bootstrap_registry
from hexai.core.config import load_config
from hexai.core.registry import registry


def main():
    """Demonstrate TOML configuration usage."""
    print("=" * 70)
    print("HexDAG TOML Configuration Example")
    print("=" * 70)
    print()

    # Example 1: Basic configuration
    print("1. BASIC CONFIGURATION")
    print("-" * 40)

    basic_config = """
# Modules to load into the registry
modules = [
    "hexai.core.ports",  # Load ports first
    "hexai.core.application.nodes",
]

plugins = [
    "hexai.adapters.mock",  # Load adapters as plugins
]

# Enable development mode
dev_mode = true
"""

    print("Config content:")
    print(basic_config)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(basic_config)
        config_path = Path(f.name)

    try:
        # Load and inspect configuration
        config = load_config(config_path)
        print(f"✓ Loaded {len(config.modules)} modules")
        print(f"✓ Dev mode: {config.dev_mode}")
        total_entries = len(config.modules) + len(config.plugins)
        print(f"✓ Configuration has {total_entries} total module entries")
        print()

        # Bootstrap registry with this config
        bootstrap_registry(config_path)
        print(f"✓ Registry bootstrapped with {len(registry.list_components())} components")
        print()
    finally:
        config_path.unlink()
        # Clean up registry
        if registry.ready:
            registry._reset_for_testing()

    # Example 2: Configuration with settings
    print("2. CONFIGURATION WITH SETTINGS")
    print("-" * 40)

    full_config = """
modules = ["hexai.core.ports", "hexai.core.application.nodes"]
plugins = ["hexai.adapters.mock"]
dev_mode = false

# Application settings
[settings]
log_level = "INFO"
enable_metrics = true
max_retries = 3

[settings.database]
connection_timeout = 30
pool_size = 10

[settings.llm]
model = "gpt-4"
temperature = 0.7
max_tokens = 1000
"""

    print("Config content (excerpt):")
    print("- Core modules and plugins")
    print("- Application settings")
    print("- Database settings")
    print("- LLM settings")
    print()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(full_config)
        config_path = Path(f.name)

    try:
        config = load_config(config_path)

        # Show settings
        print("Application settings:")
        print(f"  log_level: {config.settings.get('log_level')}")
        print(f"  enable_metrics: {config.settings.get('enable_metrics')}")
        print(f"  max_retries: {config.settings.get('max_retries')}")
        print()

        print("Database settings:")
        db_settings = config.settings.get("database", {})
        for key, value in db_settings.items():
            print(f"  {key}: {value}")
        print()

        print("LLM settings:")
        llm_settings = config.settings.get("llm", {})
        for key, value in llm_settings.items():
            print(f"  {key}: {value}")
        print()

        # Show settings
        print(f"Settings: {config.settings}")
        print()

    finally:
        config_path.unlink()

    # Example 3: Using pyproject.toml
    print("3. USING PYPROJECT.TOML")
    print("-" * 40)

    pyproject_content = """
[tool.hexdag]
modules = ["hexai.core.application.nodes"]
plugins = ["my_plugin.components"]

[tool.hexdag.settings]
log_level = "DEBUG"
enable_profiling = true

[project]
name = "my-project"
version = "1.0.0"
"""

    print("pyproject.toml with [tool.hexdag] section:")
    print(pyproject_content)

    with tempfile.NamedTemporaryFile(
        mode="w", prefix="pyproject", suffix=".toml", delete=False
    ) as f:
        f.write(pyproject_content)
        # Rename to exact pyproject.toml
        pyproject_path = Path(f.name).parent / "pyproject.toml"
        Path(f.name).rename(pyproject_path)

    try:
        config = load_config(pyproject_path)
        print("✓ Loaded config from pyproject.toml")

        print(f"✓ Modules: {config.modules}")
        print(f"✓ Plugins: {config.plugins}")
        print(f"✓ Settings: {config.settings}")
        print()
    finally:
        pyproject_path.unlink()

    print("KEY POINTS:")
    print("-" * 40)
    print("• TOML config for module and plugin management")
    print("• Supports both hexdag.toml and pyproject.toml")
    print("• Flexible settings system with nested configuration")
    print("• Dev mode for development-time component registration")
    print("• Automatic module discovery and registration")
    print("• Clean separation of core modules and plugins")


if __name__ == "__main__":
    main()
