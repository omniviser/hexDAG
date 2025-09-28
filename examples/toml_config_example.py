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
    "hexai.core.application.nodes",
    "hexai.adapters.mock",
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
        print(f"✓ Generated manifest with {len(config.get_manifest().entries)} entries")
        print()

        # Bootstrap registry with this config
        bootstrap_registry(config_path)
        print(f"✓ Registry bootstrapped with {len(registry.list_components())} components")
        print()
    finally:
        config_path.unlink()
        # Clean up registry
        if registry.ready:
            registry._components.clear()
            registry._protected_components.clear()
            registry._ready = False

    # Example 2: Configuration with bindings and environments
    print("2. CONFIGURATION WITH BINDINGS")
    print("-" * 40)

    full_config = """
modules = ["hexai.core.application.nodes"]
dev_mode = false

# Default bindings (stored for future use)
[bindings]
llm = "mock_llm"
database = { adapter = "sqlite", config = { path = "dev.db" } }

# Production environment
[env.production.bindings]
llm = "openai_adapter"
database = { adapter = "postgres", config = { host = "${DB_HOST}", port = 5432 } }

# Development environment
[env.development.bindings]
llm = "mock_llm"
database = "in_memory_db"

# Additional settings
[settings]
log_level = "INFO"
enable_metrics = true
"""

    print("Config content (excerpt):")
    print("- Default bindings defined")
    print("- Production environment config")
    print("- Development environment config")
    print("- Additional settings")
    print()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(full_config)
        config_path = Path(f.name)

    try:
        config = load_config(config_path)

        # Show default bindings
        print("Default bindings:")
        for port, binding in config.bindings.items():
            print(f"  {port}: {binding.adapter}")
        print()

        # Show environment-specific bindings
        print("Production bindings:")
        prod_bindings = config.get_bindings_for_environment("production")
        for port, binding in prod_bindings.items():
            print(f"  {port}: {binding.adapter}")
        print()

        print("Development bindings:")
        dev_bindings = config.get_bindings_for_environment("development")
        for port, binding in dev_bindings.items():
            print(f"  {port}: {binding.adapter}")
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

[tool.hexdag.bindings]
llm = "gpt4"

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
        print(f"✓ Default LLM binding: {config.bindings.get('llm', {}).adapter}")
        print()
    finally:
        pyproject_path.unlink()

    print("KEY POINTS:")
    print("-" * 40)
    print("• TOML config replaces YAML manifests")
    print("• Supports both hexdag.toml and pyproject.toml")
    print("• Environment variable substitution with ${VAR}")
    print("• Environment-specific configurations")
    print("• Bindings stored for future use (orchestrator integration in later PR)")
    print("• Manifest synthesis happens automatically")


if __name__ == "__main__":
    main()
