"""Tests for TOML configuration loader."""

import os
import tempfile
from pathlib import Path

import pytest

from hexai.core.config import ConfigLoader, config_to_manifest_entries, load_config


class TestConfigLoader:
    """Test TOML configuration loading."""

    def test_load_simple_config(self):
        """Test loading a simple TOML configuration."""
        config_content = """
modules = ["hexai.core.nodes", "hexai.core.adapters"]
plugins = ["my_plugin.components"]
dev_mode = true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            loader = ConfigLoader()
            config = loader.load_from_toml(config_path)

            assert len(config.modules) == 2
            assert "hexai.core.nodes" in config.modules
            assert len(config.plugins) == 1
            assert "my_plugin.components" in config.plugins
            assert config.dev_mode is True
        finally:
            os.unlink(config_path)

    def test_load_from_pyproject_toml(self):
        """Test loading configuration from pyproject.toml."""
        config_content = """
[tool.hexdag]
modules = ["hexai.core.nodes"]
plugins = ["external_plugin"]
"""
        # Create a file named pyproject.toml in a temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            pyproject_path = Path(tmpdir) / "pyproject.toml"
            pyproject_path.write_text(config_content)

            # Change to the temp directory for the test
            original_cwd = Path.cwd()
            try:
                os.chdir(tmpdir)

                loader = ConfigLoader()
                config = loader.load_from_toml(str(pyproject_path))

                assert len(config.modules) == 1
                assert len(config.plugins) == 1
            finally:
                os.chdir(original_cwd)

    def test_environment_variable_substitution(self):
        """Test environment variable substitution in config."""
        # Set test environment variables
        os.environ["TEST_MODULE"] = "hexai.core.test"

        config_content = """
modules = ["${TEST_MODULE}"]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            loader = ConfigLoader()
            config = loader.load_from_toml(config_path)

            assert "hexai.core.test" in config.modules
        finally:
            os.unlink(config_path)
            del os.environ["TEST_MODULE"]

    def test_manifest_entries_synthesis(self):
        """Test synthesizing manifest entries from config."""
        config_content = """
modules = [
    "hexai.core.nodes",
    "hexai.core.adapters",
    "my_app.components"
]

plugins = [
    "plugin1.components",
    "plugin2.adapters"
]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            config = load_config(config_path)
            entries = config_to_manifest_entries(config)

            assert len(entries) == 5

            # Check namespace assignment
            core_modules = [e for e in entries if e.namespace == "core"]
            assert len(core_modules) == 2  # hexai.core.nodes and hexai.core.adapters

            user_modules = [e for e in entries if e.namespace == "user"]
            assert len(user_modules) == 1  # my_app.components

            plugin_modules = [e for e in entries if e.namespace == "plugin"]
            assert len(plugin_modules) == 2  # plugin1.components and plugin2.adapters
        finally:
            os.unlink(config_path)

    def test_missing_config_file(self):
        """Test error when config file is not found."""
        loader = ConfigLoader()
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            loader.load_from_toml("/nonexistent/config.toml")

    def test_empty_config(self):
        """Test loading empty configuration."""
        config_content = ""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            config = load_config(config_path)
            assert len(config.modules) == 0
            assert len(config.plugins) == 0
            assert config.dev_mode is False
        finally:
            os.unlink(config_path)

    def test_hexdag_toml_priority(self):
        """Test that hexdag.toml takes priority over pyproject.toml."""
        # Create both files in temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create pyproject.toml
            pyproject = tmppath / "pyproject.toml"
            pyproject.write_text("""
[tool.hexdag]
modules = ["from_pyproject"]
""")

            # Create hexdag.toml
            hexdag = tmppath / "hexdag.toml"
            hexdag.write_text("""
modules = ["from_hexdag"]
""")

            # Change to temp directory
            original_cwd = Path.cwd()
            try:
                os.chdir(tmppath)
                config = load_config()
                assert config.modules == ["from_hexdag"]
            finally:
                os.chdir(original_cwd)
