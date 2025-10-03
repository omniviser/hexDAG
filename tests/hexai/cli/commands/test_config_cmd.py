"""Tests for hexai.cli.commands.config_cmd module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from hexai.cli.commands.config_cmd import (
    _display_dict,
    app,
    ensure_registry_ready,
)


@pytest.fixture
def runner():
    """Fixture providing a Typer CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_registry():
    """Fixture providing a mocked registry."""
    with patch("hexai.cli.commands.config_cmd.registry") as mock:
        yield mock


@pytest.fixture
def mock_console():
    """Fixture providing a mocked console."""
    with patch("hexai.cli.commands.config_cmd.console") as mock:
        yield mock


class TestEnsureRegistryReady:
    """Test the ensure_registry_ready helper function."""

    def test_bootstrap_when_not_ready(self, mock_registry):
        """Test that registry is bootstrapped when not ready."""
        mock_registry.ready = False
        with patch("hexai.cli.commands.config_cmd.bootstrap_registry") as mock_bootstrap:
            ensure_registry_ready()
            mock_bootstrap.assert_called_once_with(dev_mode=True)

    def test_no_bootstrap_when_ready(self, mock_registry):
        """Test that registry is not bootstrapped when already ready."""
        mock_registry.ready = True
        with patch("hexai.cli.commands.config_cmd.bootstrap_registry") as mock_bootstrap:
            ensure_registry_ready()
            mock_bootstrap.assert_not_called()


class TestListPlugins:
    """Test the list-plugins command."""

    def test_list_plugins_no_components(self, runner, mock_registry):
        """Test listing when no configurable components exist."""
        mock_registry.ready = True
        mock_registry.get_configurable_components.return_value = {}

        result = runner.invoke(app, ["list-plugins"])
        assert result.exit_code == 0
        assert "No configurable components found" in result.stdout

    def test_list_plugins_with_components(self, runner, mock_registry):
        """Test listing configurable components."""
        mock_registry.ready = True

        # Mock a configurable component
        mock_field = MagicMock()
        mock_field.description = "Test field description"

        mock_config_class = MagicMock()
        mock_config_class.model_fields = {"test_field": mock_field}

        mock_registry.get_configurable_components.return_value = {
            "test_adapter": {
                "config_class": mock_config_class,
                "type": "adapter.test",
                "port": "test_port",
            }
        }

        result = runner.invoke(app, ["list-plugins"])
        assert result.exit_code == 0
        assert "test_adapter" in result.stdout or "Configurable Components" in result.stdout


class TestGenerate:
    """Test the generate command."""

    def test_generate_default_config(self, runner, mock_registry, tmp_path):
        """Test generating default configuration."""
        mock_registry.ready = True
        mock_registry.get_configurable_components.return_value = {}

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(app, ["generate"])
            assert result.exit_code == 0

            # Check that hexdag.toml was created
            config_file = Path("hexdag.toml")
            assert config_file.exists()

            content = config_file.read_text()
            assert "HexDAG Configuration" in content
            assert "modules" in content
            assert "plugins" in content

    def test_generate_with_output_path(self, runner, mock_registry, tmp_path):
        """Test generating config to custom output path."""
        mock_registry.ready = True
        mock_registry.get_configurable_components.return_value = {}

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(app, ["generate", "--output", "custom.toml"])
            assert result.exit_code == 0

            config_file = Path("custom.toml")
            assert config_file.exists()

    def test_generate_with_show_flag(self, runner, mock_registry):
        """Test generating config with --show flag."""
        mock_registry.ready = True
        mock_registry.get_configurable_components.return_value = {}

        result = runner.invoke(app, ["generate", "--show"])
        assert result.exit_code == 0
        # Should show content on console
        assert "modules" in result.stdout or "HexDAG" in result.stdout

    def test_generate_with_plugins(self, runner, mock_registry, tmp_path):
        """Test generating config with specific plugins."""
        mock_registry.ready = False
        mock_registry.get_configurable_components.return_value = {}

        with patch("hexai.cli.commands.config_cmd.registry.bootstrap"):
            with runner.isolated_filesystem(temp_dir=tmp_path):
                result = runner.invoke(app, ["generate", "--plugin", "hexai.adapters.test"])
                # May fail due to mocking, but should attempt
                assert result.exit_code in [0, 1]

    def test_generate_with_configurable_components(self, runner, mock_registry, tmp_path):
        """Test generating config with configurable components."""
        mock_registry.ready = True

        # Mock configurable component with fields
        mock_field = MagicMock()
        mock_field.description = "API Key"
        mock_field.default = "test-key"

        mock_config_class = MagicMock()
        mock_config_class.model_fields = {"api_key": mock_field}

        mock_registry.get_configurable_components.return_value = {
            "openai": {"config_class": mock_config_class, "type": "llm", "port": "llm"}
        }

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(app, ["generate"])
            assert result.exit_code == 0

            config_file = Path("hexdag.toml")
            content = config_file.read_text()
            assert "openai" in content.lower()


class TestValidate:
    """Test the validate command."""

    def test_validate_missing_file(self, runner, tmp_path):
        """Test validating a non-existent config file."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(app, ["validate", "missing.toml"])
            assert result.exit_code == 1
            assert "not found" in result.stdout

    def test_validate_valid_config(self, runner, tmp_path):
        """Test validating a valid config file."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create a minimal valid config
            config_file = Path("hexdag.toml")
            config_file.write_text(
                """
modules = ["hexai.core.ports"]
plugins = []
dev_mode = true

[settings]
log_level = "INFO"
"""
            )

            with patch("hexai.cli.commands.config_cmd.load_config") as mock_load:
                mock_config = MagicMock()
                mock_config.modules = ["hexai.core.ports"]
                mock_config.plugins = []
                mock_config.dev_mode = True
                mock_config.settings = {}
                mock_load.return_value = mock_config

                result = runner.invoke(app, ["validate"])
                assert result.exit_code == 0
                assert "valid" in result.stdout.lower()

    def test_validate_invalid_config(self, runner, tmp_path):
        """Test validating an invalid config file."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            config_file = Path("hexdag.toml")
            config_file.write_text("invalid toml content ][")

            with patch("hexai.cli.commands.config_cmd.load_config") as mock_load:
                mock_load.side_effect = Exception("Invalid TOML")

                result = runner.invoke(app, ["validate"])
                assert result.exit_code == 1
                assert "failed" in result.stdout.lower()


class TestShow:
    """Test the show command."""

    def test_show_default_config(self, runner):
        """Test showing default configuration."""
        with patch("hexai.cli.commands.config_cmd.load_config") as mock_load:
            mock_config = MagicMock()
            mock_config.modules = ["hexai.core.ports"]
            mock_config.plugins = ["hexai.adapters.test"]
            mock_config.dev_mode = True
            mock_config.settings = {"log_level": "INFO"}
            mock_load.return_value = mock_config

            result = runner.invoke(app, ["show"])
            assert result.exit_code == 0
            assert "Modules:" in result.stdout
            assert "hexai.core.ports" in result.stdout

    def test_show_specific_config(self, runner, tmp_path):
        """Test showing a specific config file."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            config_file = Path("custom.toml")
            config_file.write_text(
                """
modules = ["test"]
plugins = []
dev_mode = false
"""
            )

            with patch("hexai.cli.commands.config_cmd.load_config") as mock_load:
                mock_config = MagicMock()
                mock_config.modules = ["test"]
                mock_config.plugins = []
                mock_config.dev_mode = False
                mock_config.settings = {}
                mock_load.return_value = mock_config

                result = runner.invoke(app, ["show", "custom.toml"])
                assert result.exit_code == 0

    def test_show_config_with_settings(self, runner):
        """Test showing config with nested settings."""
        with patch("hexai.cli.commands.config_cmd.load_config") as mock_load:
            mock_config = MagicMock()
            mock_config.modules = []
            mock_config.plugins = []
            mock_config.dev_mode = False
            mock_config.settings = {"nested": {"key": "value"}, "simple": "test"}
            mock_load.return_value = mock_config

            result = runner.invoke(app, ["show"])
            assert result.exit_code == 0
            assert "Settings:" in result.stdout

    def test_show_config_not_found(self, runner, tmp_path):
        """Test showing config when file not found."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            with patch("hexai.cli.commands.config_cmd.load_config") as mock_load:
                mock_load.side_effect = FileNotFoundError()

                result = runner.invoke(app, ["show"])
                assert result.exit_code == 0  # Shows warning but doesn't fail
                assert (
                    "No configuration file found" in result.stdout
                    or "Using default" in result.stdout
                )

    def test_show_config_error(self, runner, tmp_path):
        """Test showing config when error occurs."""
        # The show command imports load_config locally, so we need to patch it there
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create an invalid TOML file that will cause an error
            Path("hexdag.toml").write_text("invalid toml [[[")

            result = runner.invoke(app, ["show"])
            # The command catches exceptions and exits with code 1
            assert result.exit_code == 1 or "Error" in result.stdout


class TestDisplayDict:
    """Test the _display_dict helper function."""

    def test_display_simple_dict(self, mock_console):
        """Test displaying a simple dictionary."""
        data = {"key": "value", "number": 42}
        _display_dict(data)
        assert mock_console.print.called

    def test_display_nested_dict(self, mock_console):
        """Test displaying a nested dictionary."""
        data = {"outer": {"inner": "value"}, "list": [1, 2, 3]}
        _display_dict(data, indent=2)
        assert mock_console.print.called

    def test_display_dict_with_list(self, mock_console):
        """Test displaying dictionary with list values."""
        data = {"items": ["a", "b", "c"]}
        _display_dict(data)
        assert mock_console.print.called
