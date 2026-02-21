"""Tests for the config loader module.

This module tests TOML configuration loading for HexDAG.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from hexdag.core.config.loader import (
    ConfigLoader,
    _parse_bool_env,
    clear_config_cache,
    config_to_manifest_entries,
    get_default_config,
    load_config,
)
from hexdag.core.config.models import HexDAGConfig

if TYPE_CHECKING:
    from pathlib import Path


class TestParseBoolEnv:
    """Tests for _parse_bool_env function."""

    def test_truthy_values(self) -> None:
        """Test parsing truthy values."""
        for value in ["true", "True", "TRUE", "1", "yes", "YES", "on", "ON", "enabled"]:
            assert _parse_bool_env(value) is True

    def test_falsy_values(self) -> None:
        """Test parsing falsy values."""
        for value in ["false", "False", "FALSE", "0", "no", "NO", "off", "OFF", "disabled"]:
            assert _parse_bool_env(value) is False

    def test_whitespace_handling(self) -> None:
        """Test that whitespace is stripped."""
        assert _parse_bool_env("  true  ") is True
        assert _parse_bool_env("  false  ") is False

    def test_invalid_value_raises_error(self) -> None:
        """Test that invalid values raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            _parse_bool_env("maybe")
        assert "Invalid boolean value" in str(exc_info.value)
        assert "maybe" in str(exc_info.value)

    def test_empty_string_raises_error(self) -> None:
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError):
            _parse_bool_env("")


class TestConfigLoader:
    """Tests for ConfigLoader class."""

    def test_initialization(self) -> None:
        """Test ConfigLoader initialization."""
        loader = ConfigLoader()
        assert loader is not None

    def test_env_var_pattern(self) -> None:
        """Test environment variable pattern matching."""
        loader = ConfigLoader()
        # Pattern should match ${VAR_NAME}
        assert loader.ENV_VAR_PATTERN.match("${MY_VAR}")
        assert loader.ENV_VAR_PATTERN.match("${API_KEY}")
        assert not loader.ENV_VAR_PATTERN.match("$MY_VAR")
        assert not loader.ENV_VAR_PATTERN.match("MY_VAR")

    def test_substitute_env_vars_string(self) -> None:
        """Test environment variable substitution in strings."""
        loader = ConfigLoader()
        os.environ["TEST_VAR"] = "test_value"
        try:
            result = loader._substitute_env_vars("value is ${TEST_VAR}")
            assert result == "value is test_value"
        finally:
            del os.environ["TEST_VAR"]

    def test_substitute_env_vars_missing(self) -> None:
        """Test that missing env vars keep placeholder."""
        loader = ConfigLoader()
        os.environ.pop("MISSING_VAR", None)
        result = loader._substitute_env_vars("value is ${MISSING_VAR}")
        assert result == "value is ${MISSING_VAR}"

    def test_substitute_env_vars_dict(self) -> None:
        """Test environment variable substitution in dicts."""
        loader = ConfigLoader()
        os.environ["DICT_VAR"] = "dict_value"
        try:
            data = {"key": "${DICT_VAR}", "nested": {"inner": "${DICT_VAR}"}}
            result = loader._substitute_env_vars(data)
            assert result["key"] == "dict_value"
            assert result["nested"]["inner"] == "dict_value"
        finally:
            del os.environ["DICT_VAR"]

    def test_substitute_env_vars_list(self) -> None:
        """Test environment variable substitution in lists."""
        loader = ConfigLoader()
        os.environ["LIST_VAR"] = "list_value"
        try:
            data = ["${LIST_VAR}", "plain", ["${LIST_VAR}"]]
            result = loader._substitute_env_vars(data)
            assert result[0] == "list_value"
            assert result[1] == "plain"
            assert result[2][0] == "list_value"
        finally:
            del os.environ["LIST_VAR"]

    def test_substitute_env_vars_passthrough(self) -> None:
        """Test that non-string types pass through unchanged."""
        loader = ConfigLoader()
        assert loader._substitute_env_vars(123) == 123
        assert loader._substitute_env_vars(45.67) == 45.67
        assert loader._substitute_env_vars(True) is True
        assert loader._substitute_env_vars(None) is None

    def test_find_config_file_explicit_path(self, tmp_path: Path) -> None:
        """Test finding config file with explicit path."""
        loader = ConfigLoader()
        config_file = tmp_path / "test.toml"
        config_file.write_text("[tool.hexdag]\nmodules = []")

        result = loader._find_config_file(config_file)
        assert result == config_file

    def test_find_config_file_not_found(self, tmp_path: Path) -> None:
        """Test that missing explicit path raises FileNotFoundError."""
        loader = ConfigLoader()
        with pytest.raises(FileNotFoundError) as exc_info:
            loader._find_config_file(tmp_path / "nonexistent.toml")
        assert "not found" in str(exc_info.value)

    def test_find_config_file_from_env(self, tmp_path: Path) -> None:
        """Test finding config file from HEXDAG_CONFIG_PATH."""
        loader = ConfigLoader()
        config_file = tmp_path / "env_config.toml"
        config_file.write_text("[tool.hexdag]\nmodules = []")

        os.environ["HEXDAG_CONFIG_PATH"] = str(config_file)
        try:
            result = loader._find_config_file(None)
            assert result == config_file
        finally:
            del os.environ["HEXDAG_CONFIG_PATH"]

    def test_parse_config_modules(self) -> None:
        """Test parsing modules from config data."""
        loader = ConfigLoader()
        data = {"modules": ["module1", "module2"]}
        config = loader._parse_config(data)
        assert config.modules == ["module1", "module2"]

    def test_parse_config_plugins(self) -> None:
        """Test parsing plugins from config data."""
        loader = ConfigLoader()
        data = {"plugins": ["plugin1", "plugin2"]}
        config = loader._parse_config(data)
        assert config.plugins == ["plugin1", "plugin2"]

    def test_parse_config_dev_mode(self) -> None:
        """Test parsing dev_mode from config data."""
        loader = ConfigLoader()
        data = {"dev_mode": True}
        config = loader._parse_config(data)
        assert config.dev_mode is True

    def test_parse_config_settings(self) -> None:
        """Test parsing settings from config data."""
        loader = ConfigLoader()
        data = {"settings": {"key1": "value1", "key2": 42}}
        config = loader._parse_config(data)
        assert config.settings == {"key1": "value1", "key2": 42}

    def test_parse_config_empty(self) -> None:
        """Test parsing empty config data."""
        loader = ConfigLoader()
        config = loader._parse_config({})
        assert config.modules == []
        assert config.plugins == []
        assert config.dev_mode is False

    def test_parse_logging_config_defaults(self) -> None:
        """Test parsing logging config with defaults."""
        loader = ConfigLoader()
        config = loader._parse_logging_config({})
        assert config.level == "INFO"
        assert config.format == "structured"

    def test_parse_logging_config_custom(self) -> None:
        """Test parsing logging config with custom values."""
        loader = ConfigLoader()
        data = {
            "level": "DEBUG",
            "format": "json",
            "output_file": "/tmp/test.log",
            "use_color": False,
        }
        config = loader._parse_logging_config(data)
        assert config.level == "DEBUG"
        assert config.format == "json"
        assert config.output_file == "/tmp/test.log"
        assert config.use_color is False

    def test_parse_logging_config_env_override_level(self) -> None:
        """Test logging config env override for level."""
        loader = ConfigLoader()
        os.environ["HEXDAG_LOG_LEVEL"] = "ERROR"
        try:
            config = loader._parse_logging_config({"level": "INFO"})
            assert config.level == "ERROR"
        finally:
            del os.environ["HEXDAG_LOG_LEVEL"]

    def test_parse_logging_config_env_override_format(self) -> None:
        """Test logging config env override for format."""
        loader = ConfigLoader()
        os.environ["HEXDAG_LOG_FORMAT"] = "json"
        try:
            config = loader._parse_logging_config({"format": "console"})
            assert config.format == "json"
        finally:
            del os.environ["HEXDAG_LOG_FORMAT"]

    def test_parse_logging_config_env_override_file(self) -> None:
        """Test logging config env override for output_file."""
        loader = ConfigLoader()
        os.environ["HEXDAG_LOG_FILE"] = "/var/log/hexdag.log"
        try:
            config = loader._parse_logging_config({})
            assert config.output_file == "/var/log/hexdag.log"
        finally:
            del os.environ["HEXDAG_LOG_FILE"]

    def test_parse_logging_config_env_override_color(self) -> None:
        """Test logging config env override for use_color."""
        loader = ConfigLoader()
        os.environ["HEXDAG_LOG_COLOR"] = "false"
        try:
            config = loader._parse_logging_config({"use_color": True})
            assert config.use_color is False
        finally:
            del os.environ["HEXDAG_LOG_COLOR"]

    def test_parse_logging_config_env_override_timestamp(self) -> None:
        """Test logging config env override for include_timestamp."""
        loader = ConfigLoader()
        os.environ["HEXDAG_LOG_TIMESTAMP"] = "false"
        try:
            config = loader._parse_logging_config({"include_timestamp": True})
            assert config.include_timestamp is False
        finally:
            del os.environ["HEXDAG_LOG_TIMESTAMP"]

    def test_parse_logging_config_env_override_rich(self) -> None:
        """Test logging config env override for use_rich."""
        loader = ConfigLoader()
        os.environ["HEXDAG_LOG_RICH"] = "true"
        try:
            config = loader._parse_logging_config({"use_rich": False})
            assert config.use_rich is True
        finally:
            del os.environ["HEXDAG_LOG_RICH"]

    def test_parse_logging_config_env_override_dual_sink(self) -> None:
        """Test logging config env override for dual_sink."""
        loader = ConfigLoader()
        os.environ["HEXDAG_LOG_DUAL_SINK"] = "true"
        try:
            config = loader._parse_logging_config({"dual_sink": False})
            assert config.dual_sink is True
        finally:
            del os.environ["HEXDAG_LOG_DUAL_SINK"]

    def test_parse_logging_config_env_override_stdlib_bridge(self) -> None:
        """Test logging config env override for enable_stdlib_bridge."""
        loader = ConfigLoader()
        os.environ["HEXDAG_LOG_STDLIB_BRIDGE"] = "true"
        try:
            config = loader._parse_logging_config({"enable_stdlib_bridge": False})
            assert config.enable_stdlib_bridge is True
        finally:
            del os.environ["HEXDAG_LOG_STDLIB_BRIDGE"]

    def test_parse_logging_config_env_override_backtrace(self) -> None:
        """Test logging config env override for backtrace."""
        loader = ConfigLoader()
        os.environ["HEXDAG_LOG_BACKTRACE"] = "false"
        try:
            config = loader._parse_logging_config({"backtrace": True})
            assert config.backtrace is False
        finally:
            del os.environ["HEXDAG_LOG_BACKTRACE"]

    def test_parse_logging_config_env_override_diagnose(self) -> None:
        """Test logging config env override for diagnose."""
        loader = ConfigLoader()
        os.environ["HEXDAG_LOG_DIAGNOSE"] = "false"
        try:
            config = loader._parse_logging_config({"diagnose": True})
            assert config.diagnose is False
        finally:
            del os.environ["HEXDAG_LOG_DIAGNOSE"]

    def test_parse_logging_config_invalid_env_boolean(self) -> None:
        """Test that invalid boolean env var is handled gracefully."""
        loader = ConfigLoader()
        os.environ["HEXDAG_LOG_COLOR"] = "invalid"
        try:
            # Should not raise, should keep default value
            config = loader._parse_logging_config({"use_color": True})
            assert config.use_color is True  # Keeps original value
        finally:
            del os.environ["HEXDAG_LOG_COLOR"]

    def test_load_from_toml(self, tmp_path: Path) -> None:
        """Test loading config from TOML file."""
        clear_config_cache()
        config_file = tmp_path / "hexdag.toml"
        config_file.write_text("""
[tool.hexdag]
modules = ["myapp.adapters"]
plugins = ["hexdag-openai"]
dev_mode = true

[tool.hexdag.logging]
level = "DEBUG"
""")
        loader = ConfigLoader()
        config = loader.load_from_toml(config_file)
        assert config.modules == ["myapp.adapters"]
        assert config.plugins == ["hexdag-openai"]
        assert config.dev_mode is True
        assert config.logging.level == "DEBUG"

    def test_load_from_pyproject_toml(self, tmp_path: Path) -> None:
        """Test loading config from pyproject.toml."""
        clear_config_cache()
        config_file = tmp_path / "pyproject.toml"
        config_file.write_text("""
[project]
name = "myproject"

[tool.hexdag]
modules = ["myapp.nodes"]
dev_mode = false
""")
        loader = ConfigLoader()
        config = loader.load_from_toml(config_file)
        assert config.modules == ["myapp.nodes"]
        assert config.dev_mode is False

    def test_load_from_flat_toml(self, tmp_path: Path) -> None:
        """Test loading config from flat hexdag.toml (no [tool.hexdag])."""
        clear_config_cache()
        config_file = tmp_path / "hexdag.toml"
        config_file.write_text("""
modules = ["flat.module"]
dev_mode = true
""")
        loader = ConfigLoader()
        config = loader.load_from_toml(config_file)
        assert config.modules == ["flat.module"]
        assert config.dev_mode is True


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_with_path(self, tmp_path: Path) -> None:
        """Test load_config with explicit path."""
        clear_config_cache()
        config_file = tmp_path / "test.toml"
        config_file.write_text("""
modules = ["test.module"]
""")
        config = load_config(config_file)
        assert config.modules == ["test.module"]

    def test_load_config_not_found_returns_defaults(self, tmp_path: Path) -> None:
        """Test that missing config file returns defaults."""
        import pathlib

        clear_config_cache()
        # Change to temp directory where no config exists
        original_dir = pathlib.Path.cwd()
        os.chdir(tmp_path)
        try:
            config = load_config()
            # Should return default config
            assert isinstance(config, HexDAGConfig)
        finally:
            os.chdir(original_dir)


class TestGetDefaultConfig:
    """Tests for get_default_config function."""

    def test_default_config_modules(self) -> None:
        """Test default config has expected modules."""
        config = get_default_config()
        assert "hexdag.core.ports" in config.modules
        assert "hexdag.builtin.nodes" in config.modules
        assert "hexdag.builtin.adapters.mock" in config.modules
        assert "hexdag.builtin.adapters.local" in config.modules
        assert "hexdag.core.domain.agent_tools" in config.modules

    def test_default_config_settings(self) -> None:
        """Test default config has expected settings."""
        config = get_default_config()
        assert config.settings["log_level"] == "INFO"
        assert config.settings["enable_metrics"] is True


class TestClearConfigCache:
    """Tests for clear_config_cache function."""

    def test_clear_cache(self) -> None:
        """Test that clear_config_cache doesn't raise."""
        # Just verify it runs without error
        clear_config_cache()


class TestConfigToManifestEntries:
    """Tests for config_to_manifest_entries function."""

    def test_core_modules(self) -> None:
        """Test that core modules get 'core' namespace."""
        config = HexDAGConfig(
            modules=["hexdag.core.ports", "hexdag.builtin.nodes"],
        )
        entries = config_to_manifest_entries(config)
        assert len(entries) == 2
        assert all(e.namespace == "core" for e in entries)

    def test_user_modules(self) -> None:
        """Test that user modules get 'user' namespace."""
        config = HexDAGConfig(
            modules=["myapp.adapters", "myapp.nodes"],
        )
        entries = config_to_manifest_entries(config)
        assert len(entries) == 2
        assert all(e.namespace == "user" for e in entries)

    def test_plugin_modules(self) -> None:
        """Test that plugins get 'plugin' namespace."""
        config = HexDAGConfig(
            plugins=["hexdag-openai", "hexdag-postgres"],
        )
        entries = config_to_manifest_entries(config)
        assert len(entries) == 2
        assert all(e.namespace == "plugin" for e in entries)

    def test_mixed_modules(self) -> None:
        """Test mixed modules and plugins."""
        config = HexDAGConfig(
            modules=["hexdag.core.ports", "myapp.adapters"],
            plugins=["hexdag-openai"],
        )
        entries = config_to_manifest_entries(config)
        assert len(entries) == 3

        # Find entries by module
        core_entry = next(e for e in entries if e.module == "hexdag.core.ports")
        user_entry = next(e for e in entries if e.module == "myapp.adapters")
        plugin_entry = next(e for e in entries if e.module == "hexdag-openai")

        assert core_entry.namespace == "core"
        assert user_entry.namespace == "user"
        assert plugin_entry.namespace == "plugin"

    def test_empty_config(self) -> None:
        """Test empty config produces empty entries."""
        config = HexDAGConfig()
        entries = config_to_manifest_entries(config)
        assert entries == []
