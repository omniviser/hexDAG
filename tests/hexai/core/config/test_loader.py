"""Tests for TOML configuration loader."""

import os
import tempfile
from pathlib import Path

import pytest

from hexai.core.config import ConfigLoader, config_to_manifest_entries, load_config
from hexai.core.config.models import LoggingConfig


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
            Path(config_path).unlink()

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
            Path(config_path).unlink()
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
            Path(config_path).unlink()

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
            Path(config_path).unlink()

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


class TestLoggingConfigParsing:
    """Test parsing of logging configuration from TOML."""

    def test_default_logging_config(self):
        """Test that default logging config has correct values."""
        config = LoggingConfig()

        assert config.level == "INFO"
        assert config.format == "structured"
        assert config.output_file is None
        assert config.use_color is True
        assert config.include_timestamp is True

    def test_parse_logging_from_toml(self):
        """Test parsing logging section from TOML."""
        toml_content = """
[tool.hexdag.logging]
level = "DEBUG"
format = "json"
output_file = "/var/log/hexdag/app.log"
use_color = false
include_timestamp = false
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            config_path = f.name

        try:
            config = load_config(config_path)

            assert config.logging.level == "DEBUG"
            assert config.logging.format == "json"
            assert config.logging.output_file == "/var/log/hexdag/app.log"
            assert config.logging.use_color is False
            assert config.logging.include_timestamp is False
        finally:
            Path(config_path).unlink()

    def test_partial_logging_config(self):
        """Test that partial logging config uses defaults for missing fields."""
        toml_content = """
[tool.hexdag.logging]
level = "WARNING"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            config_path = f.name

        try:
            config = load_config(config_path)

            assert config.logging.level == "WARNING"
            # Defaults for other fields
            assert config.logging.format == "structured"
            assert config.logging.output_file is None
            assert config.logging.use_color is True
        finally:
            Path(config_path).unlink()

    def test_no_logging_section_uses_defaults(self):
        """Test that missing logging section uses all defaults."""
        toml_content = """
[tool.hexdag]
dev_mode = true
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            config_path = f.name

        try:
            config = load_config(config_path)

            # Should use all defaults
            assert config.logging.level == "INFO"
            assert config.logging.format == "structured"
            assert config.logging.output_file is None
        finally:
            Path(config_path).unlink()


class TestEnvironmentVariableOverrides:
    """Test environment variable overrides for logging config."""

    def setup_method(self):
        """Clear environment variables before each test."""
        env_vars = [
            "HEXDAG_LOG_LEVEL",
            "HEXDAG_LOG_FORMAT",
            "HEXDAG_LOG_FILE",
            "HEXDAG_LOG_COLOR",
            "HEXDAG_LOG_TIMESTAMP",
        ]
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]

    def test_env_level_override(self):
        """Test HEXDAG_LOG_LEVEL overrides TOML config."""
        toml_content = """
[tool.hexdag.logging]
level = "INFO"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            config_path = f.name

        try:
            os.environ["HEXDAG_LOG_LEVEL"] = "DEBUG"
            config = load_config(config_path)

            assert config.logging.level == "DEBUG"
        finally:
            Path(config_path).unlink()
            del os.environ["HEXDAG_LOG_LEVEL"]

    def test_env_format_override(self):
        """Test HEXDAG_LOG_FORMAT overrides TOML config."""
        toml_content = """
[tool.hexdag.logging]
format = "structured"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            config_path = f.name

        try:
            os.environ["HEXDAG_LOG_FORMAT"] = "json"
            config = load_config(config_path)

            assert config.logging.format == "json"
        finally:
            Path(config_path).unlink()
            del os.environ["HEXDAG_LOG_FORMAT"]

    def test_env_file_override(self):
        """Test HEXDAG_LOG_FILE overrides TOML config."""
        toml_content = """
[tool.hexdag.logging]
output_file = "/tmp/original.log"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            config_path = f.name

        try:
            os.environ["HEXDAG_LOG_FILE"] = "/tmp/override.log"
            config = load_config(config_path)

            assert config.logging.output_file == "/tmp/override.log"
        finally:
            Path(config_path).unlink()
            del os.environ["HEXDAG_LOG_FILE"]

    def test_env_color_override(self):
        """Test HEXDAG_LOG_COLOR overrides TOML config."""
        toml_content = """
[tool.hexdag.logging]
use_color = true
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            config_path = f.name

        try:
            os.environ["HEXDAG_LOG_COLOR"] = "false"
            config = load_config(config_path)

            assert config.logging.use_color is False
        finally:
            Path(config_path).unlink()
            del os.environ["HEXDAG_LOG_COLOR"]

    def test_env_timestamp_override(self):
        """Test HEXDAG_LOG_TIMESTAMP overrides TOML config."""
        toml_content = """
[tool.hexdag.logging]
include_timestamp = true
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            config_path = f.name

        try:
            os.environ["HEXDAG_LOG_TIMESTAMP"] = "0"
            config = load_config(config_path)

            assert config.logging.include_timestamp is False
        finally:
            Path(config_path).unlink()
            del os.environ["HEXDAG_LOG_TIMESTAMP"]

    def test_multiple_env_overrides(self):
        """Test multiple environment variables override together."""
        toml_content = """
[tool.hexdag.logging]
level = "INFO"
format = "console"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            config_path = f.name

        try:
            os.environ["HEXDAG_LOG_LEVEL"] = "ERROR"
            os.environ["HEXDAG_LOG_FORMAT"] = "json"
            os.environ["HEXDAG_LOG_FILE"] = "/tmp/test.log"

            config = load_config(config_path)

            assert config.logging.level == "ERROR"
            assert config.logging.format == "json"
            assert config.logging.output_file == "/tmp/test.log"
        finally:
            Path(config_path).unlink()
            del os.environ["HEXDAG_LOG_LEVEL"]
            del os.environ["HEXDAG_LOG_FORMAT"]
            del os.environ["HEXDAG_LOG_FILE"]

    def test_env_without_toml_section(self):
        """Test env vars work even when TOML has no logging section."""
        toml_content = """
[tool.hexdag]
dev_mode = true
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            config_path = f.name

        try:
            os.environ["HEXDAG_LOG_LEVEL"] = "DEBUG"
            config = load_config(config_path)

            assert config.logging.level == "DEBUG"
        finally:
            Path(config_path).unlink()
            del os.environ["HEXDAG_LOG_LEVEL"]


class TestBootstrapIntegration:
    """Test integration with bootstrap process."""

    def test_bootstrap_configures_logging(self):
        """Test that bootstrap_registry() configures logging from config."""
        from hexai.core.bootstrap import bootstrap_registry
        from hexai.core.logging import get_logger
        from hexai.core.registry import registry

        toml_content = """
[tool.hexdag.logging]
level = "DEBUG"
format = "console"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            config_path = f.name

        try:
            # Reset registry
            if registry.ready:
                registry._ready = False
                registry._components.clear()

            # Bootstrap with config
            bootstrap_registry(config_path)

            # Verify we can get a logger and it works
            logger = get_logger("test")
            logger.debug("Test message")  # Should not raise

            assert registry.ready
        finally:
            Path(config_path).unlink()
            # Reset for other tests
            if registry.ready:
                registry._ready = False
                registry._components.clear()
