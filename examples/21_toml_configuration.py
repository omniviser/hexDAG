"""Example 21: TOML Configuration for HexDAG.

This example demonstrates:
- Loading configuration from TOML files
- Module and plugin management
- Environment-specific configurations
- Manifest synthesis from TOML
"""

import asyncio
import os
import tempfile
from pathlib import Path

from hexdag.core.bootstrap import bootstrap_registry
from hexdag.core.config import config_to_manifest_entries, load_config
from hexdag.core.registry import registry


async def main() -> None:
    """Demonstrate TOML configuration for HexDAG."""
    print("🔧 Example 21: TOML Configuration")
    print("=" * 60)
    print()

    # Example 1: Basic TOML Configuration
    print("📄 Test 1: Basic TOML Configuration")
    print("-" * 50)

    basic_config = """
# Core modules to load
modules = [
    "hexdag.builtin.nodes",
]

# Enable development mode
dev_mode = true

# Additional settings
[settings]
log_level = "INFO"
enable_metrics = true
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(basic_config)
        config_path = Path(f.name)

    try:
        # Load configuration
        config = load_config(config_path)
        print("📋 Configuration loaded:")
        print(f"   ✓ Modules: {len(config.modules)} module(s)")
        print(f"   ✓ Dev mode: {config.dev_mode}")
        print(f"   ✓ Settings: {len(config.settings)} setting(s)")

        # Generate manifest entries
        entries = config_to_manifest_entries(config)
        print(f"   ✓ Generated entries: {len(entries)} entries")

        # Bootstrap registry
        bootstrap_registry(config_path)
        print(f"   ✓ Registry bootstrapped: {len(registry.list_components())} components")
        print()
    finally:
        config_path.unlink()
        # Clean up registry for next example
        if registry.ready:
            registry._reset_for_testing()

    # Example 2: Module and Plugin Management
    print("📦 Test 2: Module and Plugin Management")
    print("-" * 50)

    module_config = """
# Core modules
modules = [
    "hexdag.builtin.nodes",
    "hexdag.core.adapters",
    "my_app.components",
]

# Third-party plugins
plugins = [
    "llm_plugin.adapters",
    "monitoring_plugin.observers",
]

# Settings
[settings]
log_level = "DEBUG"
max_workers = 10
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(module_config)
        config_path = Path(f.name)

    try:
        config = load_config(config_path)

        print("📚 Modules:")
        for module in config.modules:
            print(f"   • {module}")
        print()

        print("🔌 Plugins:")
        for plugin in config.plugins:
            print(f"   • {plugin}")
        print()

        print("⚙️ Settings:")
        if config.settings:
            for key, value in config.settings.items():
                print(f"   • {key}: {value}")
        else:
            print("   • (none)")
        print()

        # Show namespace assignment
        entries = config_to_manifest_entries(config)
        print("📂 Namespace assignments:")
        namespaces = {}
        for entry in entries:
            if entry.namespace not in namespaces:
                namespaces[entry.namespace] = []
            namespaces[entry.namespace].append(entry.module)

        for ns, modules in namespaces.items():
            print(f"   • {ns}: {', '.join(modules)}")
        print()
    finally:
        config_path.unlink()

    # Example 3: Environment Variable Substitution
    print("🔄 Test 3: Environment Variable Substitution")
    print("-" * 50)

    # Set test environment variables
    os.environ["TEST_MODULE"] = "my_custom.module"
    os.environ["TEST_LOG_LEVEL"] = "WARNING"

    var_config = """
modules = [
    "hexdag.builtin.nodes",
    "${TEST_MODULE}",
]

[settings]
log_level = "${TEST_LOG_LEVEL}"
api_endpoint = "https://api.example.com"
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(var_config)
        config_path = Path(f.name)

    try:
        config = load_config(config_path)

        print("🔑 Environment variables substituted:")
        print(f"   • Custom module: {config.modules[1]}")
        print(f"   • Log level: {config.settings.get('log_level', 'N/A')}")
        print(f"   • API endpoint: {config.settings.get('api_endpoint', 'N/A')}")
        print()
    finally:
        config_path.unlink()
        # Clean up env vars
        del os.environ["TEST_MODULE"]
        del os.environ["TEST_LOG_LEVEL"]

    # Example 4: Using pyproject.toml
    print("📦 Test 4: Using pyproject.toml")
    print("-" * 50)

    # Check if we're in the project root with pyproject.toml
    pyproject_path = Path("pyproject.toml")
    if pyproject_path.exists():
        try:
            config = load_config(pyproject_path)
            print("✓ Loaded configuration from pyproject.toml")
            print(f"   • Modules: {config.modules[:2]}...")  # Show first 2
            print(f"   • Plugins: {config.plugins if config.plugins else 'None'}")
        except Exception as e:
            print(f"   ⚠️ Could not load [tool.hexdag] from pyproject.toml: {e}")
    else:
        print("   ℹ️ No pyproject.toml found in current directory")
    print()

    # Example 5: Settings and Additional Configuration
    print("⚙️ Test 5: Settings and Additional Configuration")
    print("-" * 50)

    settings_config = """
modules = ["hexdag.builtin.nodes"]

dev_mode = false

[settings]
# Logging configuration
log_level = "INFO"
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Performance settings
max_workers = 8
timeout_seconds = 30
enable_profiling = true

# Feature flags
enable_metrics = true
enable_tracing = false
experimental_features = ["async_validation", "parallel_bootstrap"]
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(settings_config)
        config_path = Path(f.name)

    try:
        config = load_config(config_path)

        print("📊 Configuration settings:")
        print(f"   • Dev mode: {config.dev_mode}")
        print()

        print("📝 Logging:")
        print(f"   • Level: {config.settings.get('log_level', 'INFO')}")
        log_format = config.settings.get("log_format")
        if log_format:
            print(f"   • Format: {log_format[:30]}...")
        else:
            print("   • Format: (default)")

        print()

        print("⚡ Performance:")
        print(f"   • Max workers: {config.settings.get('max_workers', 4)}")
        timeout = config.settings.get("timeout_seconds")
        print(f"   • Timeout: {timeout}s" if timeout else "   • Timeout: (default)")
        print(f"   • Profiling: {config.settings.get('enable_profiling', False)}")
        print()

        print("🚀 Feature flags:")
        print(f"   • Metrics: {config.settings.get('enable_metrics', False)}")
        print(f"   • Tracing: {config.settings.get('enable_tracing', False)}")
        experimental = config.settings.get("experimental_features", [])
        if experimental:
            print(f"   • Experimental: {', '.join(experimental)}")
        print()
    finally:
        config_path.unlink()

    # Summary
    print("📈 Configuration Features Summary")
    print("-" * 50)
    print("✅ TOML format replaces YAML manifests")
    print("✅ Support for both hexdag.toml and pyproject.toml")
    print("✅ Environment variable substitution with ${VAR}")
    print("✅ Module and plugin management")
    print("✅ Flexible settings configuration")
    print("✅ Automatic namespace assignment")
    print()

    print("🎯 Key Benefits:")
    print("   • Standard TOML format")
    print("   • Better IDE support")
    print("   • Single configuration source")
    print("   • Clear separation of modules and plugins")
    print("   • Extensible settings system")
    print()


if __name__ == "__main__":
    asyncio.run(main())
