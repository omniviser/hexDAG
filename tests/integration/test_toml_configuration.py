"""Integration tests for TOML configuration.

Tests demonstrate:
- Loading configuration from TOML files
- Module and plugin management
- Environment-specific configurations
- Configuration validation
"""

import os
import tempfile
from pathlib import Path

import pytest

from hexdag.core.bootstrap import bootstrap_registry
from hexdag.core.config import config_to_manifest_entries, load_config
from hexdag.core.registry import registry


@pytest.fixture(autouse=True)
def cleanup_registry():
    """Clean up registry before and after each test."""
    if registry.ready:
        registry._reset_for_testing()
    yield
    if registry.ready:
        registry._reset_for_testing()


class TestTOMLConfiguration:
    """Test suite for TOML configuration functionality."""

    def test_basic_toml_loading(self):
        """Test loading basic TOML configuration."""
        config_content = """
modules = ["hexdag.builtin.nodes"]
dev_mode = true

[settings]
log_level = "INFO"
enable_metrics = true
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_content)
            config_path = Path(f.name)

        try:
            config = load_config(config_path)

            assert len(config.modules) == 1
            assert config.dev_mode is True
            assert config.settings.get("log_level") == "INFO"
            assert config.settings.get("enable_metrics") is True
        finally:
            config_path.unlink()

    def test_modules_and_plugins(self):
        """Test module and plugin configuration."""
        config_content = """
modules = [
    "hexdag.builtin.nodes",
    "hexdag.core.adapters",
]

plugins = [
    "llm_plugin.adapters",
]

[settings]
log_level = "DEBUG"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_content)
            config_path = Path(f.name)

        try:
            config = load_config(config_path)

            assert len(config.modules) == 2
            assert len(config.plugins) == 1
            assert "hexdag.builtin.nodes" in config.modules
            assert "llm_plugin.adapters" in config.plugins
        finally:
            config_path.unlink()

    def test_environment_variable_substitution(self):
        """Test environment variable substitution in TOML."""
        os.environ["TEST_MODULE"] = "my_custom.module"
        os.environ["TEST_LOG_LEVEL"] = "WARNING"

        config_content = """
modules = [
    "hexdag.builtin.nodes",
    "${TEST_MODULE}",
]

[settings]
log_level = "${TEST_LOG_LEVEL}"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_content)
            config_path = Path(f.name)

        try:
            config = load_config(config_path)

            assert config.modules[1] == "my_custom.module"
            assert config.settings.get("log_level") == "WARNING"
        finally:
            config_path.unlink()
            del os.environ["TEST_MODULE"]
            del os.environ["TEST_LOG_LEVEL"]

    def test_manifest_entry_generation(self):
        """Test generating manifest entries from config."""
        config_content = """
modules = [
    "hexdag.builtin.nodes",
    "hexdag.builtin.adapters",
]
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_content)
            config_path = Path(f.name)

        try:
            config = load_config(config_path)
            entries = config_to_manifest_entries(config)

            assert len(entries) >= 2
            # Check that entries have required fields
            for entry in entries:
                assert hasattr(entry, "module")
                assert hasattr(entry, "namespace")
        finally:
            config_path.unlink()

    def test_settings_section(self):
        """Test various settings configurations."""
        config_content = """
modules = ["hexdag.builtin.nodes"]

[settings]
log_level = "DEBUG"
max_workers = 10
timeout_seconds = 30
enable_profiling = true
experimental_features = ["async_validation", "parallel_bootstrap"]
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_content)
            config_path = Path(f.name)

        try:
            config = load_config(config_path)

            assert config.settings["log_level"] == "DEBUG"
            assert config.settings["max_workers"] == 10
            assert config.settings["timeout_seconds"] == 30
            assert config.settings["enable_profiling"] is True
            assert len(config.settings["experimental_features"]) == 2
        finally:
            config_path.unlink()

    def test_empty_configuration(self):
        """Test loading empty/minimal configuration."""
        config_content = """
modules = []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_content)
            config_path = Path(f.name)

        try:
            config = load_config(config_path)

            assert len(config.modules) == 0
            assert config.plugins == []
        finally:
            config_path.unlink()

    def test_bootstrap_with_config(self):
        """Test bootstrapping registry with TOML config."""
        config_content = """
modules = ["hexdag.builtin.nodes"]
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_content)
            config_path = Path(f.name)

        try:
            bootstrap_registry(config_path)

            # Registry should be ready
            assert registry.ready
            # Should have loaded components
            components = registry.list_components()
            assert len(components) > 0
        finally:
            config_path.unlink()
