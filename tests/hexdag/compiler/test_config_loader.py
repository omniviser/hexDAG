"""Tests for the config loader module (compiler location).

This module tests configuration loading for HexDAG, covering both
kind: Config YAML manifests and pyproject.toml [tool.hexdag].
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from hexdag.compiler.config_loader import (
    ConfigLoader,
    _parse_bool_env,
    clear_config_cache,
    config_to_manifest_entries,
    get_default_config,
    load_config,
)
from hexdag.kernel.config.models import HexDAGConfig

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
        config_file = tmp_path / "test.yaml"
        config_file.write_text("kind: Config\nmetadata:\n  name: test\nspec: {}")

        result = loader._find_config_file(config_file)
        assert result == config_file

    def test_find_config_file_not_found(self, tmp_path: Path) -> None:
        """Test that missing explicit path raises FileNotFoundError."""
        loader = ConfigLoader()
        with pytest.raises(FileNotFoundError) as exc_info:
            loader._find_config_file(tmp_path / "nonexistent.yaml")
        assert "not found" in str(exc_info.value)

    def test_find_config_file_from_env(self, tmp_path: Path) -> None:
        """Test finding config file from HEXDAG_CONFIG_PATH."""
        loader = ConfigLoader()
        config_file = tmp_path / "env_config.yaml"
        config_file.write_text("kind: Config\nmetadata:\n  name: test\nspec: {}")

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
        config = loader.load_config_file(config_file)
        assert config.modules == ["myapp.nodes"]
        assert config.dev_mode is False


class TestYamlConfigLoading:
    """Tests for YAML kind: Config file loading."""

    def test_load_yaml_config_full_manifest(self, tmp_path: Path) -> None:
        """Test loading full kind: Config YAML manifest."""
        clear_config_cache()
        config_file = tmp_path / "hexdag.yaml"
        config_file.write_text("""\
kind: Config
metadata:
  name: test-config
spec:
  modules:
    - myapp.adapters
  plugins:
    - hexdag-openai
  logging:
    level: DEBUG
  kernel:
    max_concurrent_nodes: 5
    default_node_timeout: 60.0
  limits:
    max_llm_calls: 100
    max_cost_usd: 10.0
  caps:
    default_set:
      - llm
      - memory
    deny:
      - secret
""")
        config = load_config(config_file)
        assert config.modules == ["myapp.adapters"]
        assert config.plugins == ["hexdag-openai"]
        assert config.logging.level == "DEBUG"
        assert config.orchestrator.max_concurrent_nodes == 5
        assert config.orchestrator.default_node_timeout == 60.0
        assert config.limits.max_llm_calls == 100
        assert config.limits.max_cost_usd == 10.0
        assert config.caps.default_set == ["llm", "memory"]
        assert config.caps.deny == ["secret"]

    def test_yaml_env_var_substitution(self, tmp_path: Path) -> None:
        """Test env var substitution works in YAML config."""
        clear_config_cache()
        os.environ["TEST_MODULE"] = "myapp.custom"
        try:
            config_file = tmp_path / "config.yaml"
            config_file.write_text("""\
kind: Config
metadata:
  name: env-test
spec:
  modules:
    - ${TEST_MODULE}
""")
            config = load_config(config_file)
            assert config.modules == ["myapp.custom"]
        finally:
            del os.environ["TEST_MODULE"]

    def test_yaml_config_missing_kind_raises(self, tmp_path: Path) -> None:
        """Test YAML without kind: Config raises error."""
        clear_config_cache()
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("modules:\n  - foo\n")
        with pytest.raises(ValueError, match="kind: Config"):
            loader = ConfigLoader()
            loader.load_config_file(config_file)

    def test_yaml_config_wrong_kind_raises(self, tmp_path: Path) -> None:
        """Test kind: Pipeline as standalone config raises error."""
        clear_config_cache()
        config_file = tmp_path / "wrong.yaml"
        config_file.write_text("""\
kind: Pipeline
metadata:
  name: not-a-config
spec:
  nodes: []
""")
        with pytest.raises(ValueError, match="kind: Config"):
            loader = ConfigLoader()
            loader.load_config_file(config_file)

    def test_yaml_config_invalid_yaml_raises(self, tmp_path: Path) -> None:
        """Test that non-dict YAML raises error."""
        clear_config_cache()
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("- just\n- a\n- list\n")
        with pytest.raises(ValueError, match="expected a mapping"):
            loader = ConfigLoader()
            loader.load_config_file(config_file)

    def test_yaml_config_empty_spec(self, tmp_path: Path) -> None:
        """Test kind: Config with empty spec returns default-like config."""
        clear_config_cache()
        config_file = tmp_path / "minimal.yaml"
        config_file.write_text("""\
kind: Config
metadata:
  name: minimal
spec: {}
""")
        config = load_config(config_file)
        assert config.modules == []
        assert config.dev_mode is False

    def test_explicit_path_loads_yaml(self, tmp_path: Path) -> None:
        """Test explicit path argument pointing to .yaml file."""
        clear_config_cache()
        config_file = tmp_path / "custom-name.yaml"
        config_file.write_text("""\
kind: Config
metadata:
  name: custom
spec:
  dev_mode: true
""")
        config = load_config(config_file)
        assert config.dev_mode is True

    def test_env_var_path_loads_yaml(self, tmp_path: Path) -> None:
        """Test HEXDAG_CONFIG_PATH env var pointing to YAML file."""
        clear_config_cache()
        config_file = tmp_path / "env-config.yaml"
        config_file.write_text("""\
kind: Config
metadata:
  name: from-env
spec:
  modules:
    - env.module
""")
        os.environ["HEXDAG_CONFIG_PATH"] = str(config_file)
        try:
            config = load_config()
            assert config.modules == ["env.module"]
        finally:
            del os.environ["HEXDAG_CONFIG_PATH"]


class TestConfigFileDiscovery:
    """Tests for config file discovery order."""

    def test_auto_discovery_does_not_find_hexdag_toml(self, tmp_path: Path) -> None:
        """Test that auto-discovery does NOT find hexdag.toml."""
        import pathlib

        clear_config_cache()
        # Create hexdag.toml - should NOT be discovered
        (tmp_path / "hexdag.toml").write_text('modules = ["toml.mod"]')

        original_dir = pathlib.Path.cwd()
        os.chdir(tmp_path)
        try:
            loader = ConfigLoader()
            with pytest.raises(FileNotFoundError):
                loader._find_config_file(None)
        finally:
            os.chdir(original_dir)

    def test_auto_discovery_finds_pyproject(self, tmp_path: Path) -> None:
        """Test that auto-discovery finds pyproject.toml."""
        import pathlib

        clear_config_cache()
        (tmp_path / "pyproject.toml").write_text("""
[tool.hexdag]
modules = ["pyproject.mod"]
""")

        original_dir = pathlib.Path.cwd()
        os.chdir(tmp_path)
        try:
            loader = ConfigLoader()
            result = loader._find_config_file(None)
            assert result.name == "pyproject.toml"
        finally:
            os.chdir(original_dir)

    def test_auto_discovery_finds_hexdag_yaml(self, tmp_path: Path) -> None:
        """Test that auto-discovery finds hexdag.yaml in CWD."""
        import pathlib

        clear_config_cache()
        (tmp_path / "hexdag.yaml").write_text("""\
kind: Config
metadata:
  name: test
spec:
  modules:
    - yaml.mod
""")

        original_dir = pathlib.Path.cwd()
        os.chdir(tmp_path)
        try:
            loader = ConfigLoader()
            result = loader._find_config_file(None)
            assert result.name == "hexdag.yaml"
        finally:
            os.chdir(original_dir)

    def test_auto_discovery_finds_hexdag_yml(self, tmp_path: Path) -> None:
        """Test that auto-discovery finds hexdag.yml in CWD."""
        import pathlib

        clear_config_cache()
        (tmp_path / "hexdag.yml").write_text("""\
kind: Config
metadata:
  name: test
spec:
  modules:
    - yml.mod
""")

        original_dir = pathlib.Path.cwd()
        os.chdir(tmp_path)
        try:
            loader = ConfigLoader()
            result = loader._find_config_file(None)
            assert result.name == "hexdag.yml"
        finally:
            os.chdir(original_dir)

    def test_yaml_takes_priority_over_pyproject(self, tmp_path: Path) -> None:
        """Test that hexdag.yaml takes priority over pyproject.toml."""
        import pathlib

        clear_config_cache()
        (tmp_path / "hexdag.yaml").write_text("""\
kind: Config
metadata:
  name: yaml-config
spec:
  modules:
    - yaml.mod
""")
        (tmp_path / "pyproject.toml").write_text("""
[tool.hexdag]
modules = ["pyproject.mod"]
""")

        original_dir = pathlib.Path.cwd()
        os.chdir(tmp_path)
        try:
            loader = ConfigLoader()
            result = loader._find_config_file(None)
            assert result.name == "hexdag.yaml"
        finally:
            os.chdir(original_dir)

    def test_parent_directory_traversal_finds_hexdag_yaml(self, tmp_path: Path) -> None:
        """Test that parent directory traversal finds hexdag.yaml."""
        import pathlib

        clear_config_cache()
        # Create hexdag.yaml in parent
        (tmp_path / "hexdag.yaml").write_text("""\
kind: Config
metadata:
  name: parent-config
spec:
  modules:
    - parent.mod
""")
        # Create a child directory
        child_dir = tmp_path / "subdir"
        child_dir.mkdir()

        original_dir = pathlib.Path.cwd()
        os.chdir(child_dir)
        try:
            loader = ConfigLoader()
            result = loader._find_config_file(None)
            assert result.name == "hexdag.yaml"
            assert result.parent == tmp_path
        finally:
            os.chdir(original_dir)

    def test_toml_loading_emits_deprecation_warning(self, tmp_path: Path) -> None:
        """Test that loading from TOML emits a deprecation warning."""
        import warnings

        clear_config_cache()
        config_file = tmp_path / "pyproject.toml"
        config_file.write_text("""
[tool.hexdag]
modules = ["test.mod"]
""")
        loader = ConfigLoader()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loader.load_config_file(config_file)
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1
            assert "TOML is deprecated" in str(deprecation_warnings[0].message)


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_with_path(self, tmp_path: Path) -> None:
        """Test load_config with explicit path."""
        clear_config_cache()
        config_file = tmp_path / "test.yaml"
        config_file.write_text("""\
kind: Config
metadata:
  name: test
spec:
  modules:
    - test.module
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
        assert "hexdag.kernel.ports" in config.modules
        assert "hexdag.stdlib.nodes" in config.modules
        assert "hexdag.stdlib.adapters.mock" in config.modules
        assert "hexdag.drivers.executors" in config.modules
        assert "hexdag.drivers.observer_manager" in config.modules
        assert "hexdag.drivers.pipeline_spawner" in config.modules
        assert "hexdag.kernel.domain.agent_tools" in config.modules

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
            modules=["hexdag.kernel.ports", "hexdag.stdlib.nodes"],
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
            modules=["hexdag.kernel.ports", "myapp.adapters"],
            plugins=["hexdag-openai"],
        )
        entries = config_to_manifest_entries(config)
        assert len(entries) == 3

        # Find entries by module
        core_entry = next(e for e in entries if e.module == "hexdag.kernel.ports")
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


class TestParseConfigOrchestratorSection:
    """Tests for parsing orchestrator/kernel config section."""

    def test_parse_orchestrator_section(self) -> None:
        """Test parsing orchestrator section."""
        loader = ConfigLoader()
        data = {
            "orchestrator": {
                "max_concurrent_nodes": 5,
                "strict_validation": True,
                "default_node_timeout": 60.0,
            },
        }
        config = loader._parse_config(data)
        assert config.orchestrator.max_concurrent_nodes == 5
        assert config.orchestrator.strict_validation is True
        assert config.orchestrator.default_node_timeout == 60.0

    def test_parse_kernel_alias(self) -> None:
        """Test that 'kernel' key works as alias for 'orchestrator'."""
        loader = ConfigLoader()
        data = {
            "kernel": {
                "max_concurrent_nodes": 3,
                "default_node_timeout": 30.0,
            },
        }
        config = loader._parse_config(data)
        assert config.orchestrator.max_concurrent_nodes == 3
        assert config.orchestrator.default_node_timeout == 30.0

    def test_orchestrator_takes_precedence_over_kernel(self) -> None:
        """Test that 'orchestrator' key takes precedence over 'kernel'."""
        loader = ConfigLoader()
        data = {
            "orchestrator": {"max_concurrent_nodes": 5},
            "kernel": {"max_concurrent_nodes": 3},
        }
        config = loader._parse_config(data)
        assert config.orchestrator.max_concurrent_nodes == 5

    def test_no_orchestrator_uses_defaults(self) -> None:
        """Test that missing orchestrator section uses defaults."""
        loader = ConfigLoader()
        config = loader._parse_config({})
        assert config.orchestrator.max_concurrent_nodes == 10
        assert config.orchestrator.strict_validation is False
        assert config.orchestrator.default_node_timeout is None


class TestParseConfigLimitsSection:
    """Tests for parsing limits config section."""

    def test_parse_full_limits(self) -> None:
        """Test parsing full limits section."""
        loader = ConfigLoader()
        data = {
            "limits": {
                "max_total_tokens": 100000,
                "max_llm_calls": 50,
                "max_tool_calls": 200,
                "max_cost_usd": 5.0,
                "warning_threshold": 0.9,
            },
        }
        config = loader._parse_config(data)
        assert config.limits.max_total_tokens == 100000
        assert config.limits.max_llm_calls == 50
        assert config.limits.max_tool_calls == 200
        assert config.limits.max_cost_usd == 5.0
        assert config.limits.warning_threshold == 0.9

    def test_parse_partial_limits(self) -> None:
        """Test parsing partial limits section."""
        loader = ConfigLoader()
        data = {
            "limits": {
                "max_llm_calls": 100,
                "max_cost_usd": 10.0,
            },
        }
        config = loader._parse_config(data)
        assert config.limits.max_llm_calls == 100
        assert config.limits.max_cost_usd == 10.0
        assert config.limits.max_total_tokens is None
        assert config.limits.max_tool_calls is None
        assert config.limits.warning_threshold == 0.8

    def test_no_limits_uses_defaults(self) -> None:
        """Test that missing limits section uses defaults."""
        loader = ConfigLoader()
        config = loader._parse_config({})
        assert config.limits.max_llm_calls is None
        assert config.limits.max_cost_usd is None


class TestParseConfigCapsSection:
    """Tests for parsing caps config section."""

    def test_parse_full_caps(self) -> None:
        """Test parsing full caps section."""
        loader = ConfigLoader()
        data = {
            "caps": {
                "default_set": ["llm", "memory", "datastore.read"],
                "deny": ["secret", "spawner"],
            },
        }
        config = loader._parse_config(data)
        assert config.caps.default_set == ["llm", "memory", "datastore.read"]
        assert config.caps.deny == ["secret", "spawner"]

    def test_parse_deny_only(self) -> None:
        """Test parsing caps with deny only."""
        loader = ConfigLoader()
        data = {
            "caps": {
                "deny": ["secret"],
            },
        }
        config = loader._parse_config(data)
        assert config.caps.default_set is None
        assert config.caps.deny == ["secret"]

    def test_no_caps_uses_defaults(self) -> None:
        """Test that missing caps section uses defaults (unrestricted)."""
        loader = ConfigLoader()
        config = loader._parse_config({})
        assert config.caps.default_set is None
        assert config.caps.deny is None

    def test_load_yaml_with_all_sections(self, tmp_path: Path) -> None:
        """Test loading YAML file with orchestrator, limits, and caps sections."""
        clear_config_cache()
        config_file = tmp_path / "full.yaml"
        config_file.write_text("""\
kind: Config
metadata:
  name: full-config
spec:
  modules:
    - myapp.nodes
  kernel:
    max_concurrent_nodes: 5
    default_node_timeout: 60.0
  limits:
    max_llm_calls: 100
    max_cost_usd: 10.0
  caps:
    deny:
      - secret
""")
        loader = ConfigLoader()
        config = loader.load_config_file(config_file)
        assert config.modules == ["myapp.nodes"]
        assert config.orchestrator.max_concurrent_nodes == 5
        assert config.orchestrator.default_node_timeout == 60.0
        assert config.limits.max_llm_calls == 100
        assert config.limits.max_cost_usd == 10.0
        assert config.caps.deny == ["secret"]
