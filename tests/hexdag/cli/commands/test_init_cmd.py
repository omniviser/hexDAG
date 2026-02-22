"""Tests for hexdag.cli.commands.init_cmd module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from hexdag.cli.commands.init_cmd import _generate_config, _print_adapter_info, app, init


@pytest.fixture
def runner():
    """Fixture providing a Typer CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_console():
    """Fixture providing a mocked console."""
    with patch("hexdag.cli.commands.init_cmd.console") as mock:
        yield mock


class TestInit:
    """Test the init command."""

    def test_init_creates_config_in_current_dir(self, runner, tmp_path):
        """Test init creates hexdag.toml in current directory."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(app, [])
            assert result.exit_code == 0

            config_file = Path("hexdag.toml")
            assert config_file.exists()

            content = config_file.read_text()
            assert "HexDAG Configuration" in content
            assert "hexdag.kernel.ports" in content

    def test_init_creates_config_in_specified_dir(self, runner, tmp_path):
        """Test init creates hexdag.toml in specified directory."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            test_dir = Path("my_project")
            result = runner.invoke(app, [str(test_dir)])
            assert result.exit_code == 0

            config_file = test_dir / "hexdag.toml"
            assert config_file.exists()

    def test_init_with_adapters(self, runner, tmp_path):
        """Test init with adapter list."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(app, ["--with", "openai,anthropic"])
            assert result.exit_code == 0

            config_file = Path("hexdag.toml")
            content = config_file.read_text()
            assert "openai" in content.lower()
            assert "anthropic" in content.lower()

    def test_init_with_force_flag(self, runner, tmp_path):
        """Test init --force overwrites existing config."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create existing config
            config_file = Path("hexdag.toml")
            config_file.write_text("# Old config")

            result = runner.invoke(app, ["--force"])
            assert result.exit_code == 0

            # Should be overwritten
            content = config_file.read_text()
            assert "HexDAG Configuration" in content
            assert "Old config" not in content

    def test_init_refuses_overwrite_without_force(self, runner, tmp_path):
        """Test init refuses to overwrite without --force."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create existing config
            config_file = Path("hexdag.toml")
            config_file.write_text("# Old config")

            # Simulate user declining overwrite
            result = runner.invoke(app, [], input="n\n")
            assert result.exit_code == 1
            assert "cancelled" in result.stdout.lower()

    def test_init_accepts_overwrite_with_confirmation(self, runner, tmp_path):
        """Test init overwrites when user confirms."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create existing config
            config_file = Path("hexdag.toml")
            config_file.write_text("# Old config")

            # Simulate user confirming overwrite
            result = runner.invoke(app, [], input="y\n")
            assert result.exit_code == 0

            content = config_file.read_text()
            assert "HexDAG Configuration" in content

    def test_init_creates_directory_if_not_exists(self, runner, tmp_path):
        """Test init creates directory if it doesn't exist."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            nested_dir = Path("nested/project/dir")
            result = runner.invoke(app, [str(nested_dir)])
            assert result.exit_code == 0

            assert nested_dir.exists()
            assert (nested_dir / "hexdag.toml").exists()

    def test_init_shows_next_steps(self, runner, tmp_path):
        """Test init shows helpful next steps."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(app, [])
            assert result.exit_code == 0
            assert "Next steps:" in result.stdout
            assert "hexdag.toml" in result.stdout

    def test_init_with_context_invoked_subcommand(self, tmp_path):
        """Test init callback returns early when subcommand is invoked."""
        ctx = MagicMock()
        ctx.invoked_subcommand = "some_subcommand"

        # Should return None immediately without doing anything
        result = init(ctx=ctx, with_adapters=None, force=False, path=None)
        assert result is None


class TestGenerateConfig:
    """Test the _generate_config helper function."""

    def test_generate_config_no_adapters(self):
        """Test generating config without adapters."""
        config = _generate_config([])
        assert "HexDAG Configuration" in config
        assert "hexdag.kernel.ports" in config
        assert "hexdag.drivers.mock" in config

    def test_generate_config_with_openai(self):
        """Test generating config with OpenAI adapter."""
        config = _generate_config(["openai"])
        assert "openai" in config.lower()
        assert "OPENAI_API_KEY" in config

    def test_generate_config_with_anthropic(self):
        """Test generating config with Anthropic adapter."""
        config = _generate_config(["anthropic"])
        assert "anthropic" in config.lower()
        assert "ANTHROPIC_API_KEY" in config

    def test_generate_config_with_local(self):
        """Test generating config with local adapter."""
        config = _generate_config(["local"])
        assert "local" in config.lower()
        assert "memory_max_size" in config

    def test_generate_config_with_multiple_adapters(self):
        """Test generating config with multiple adapters."""
        config = _generate_config(["openai", "anthropic", "local"])
        assert "openai" in config.lower()
        assert "anthropic" in config.lower()
        assert "local" in config.lower()

    def test_generate_config_structure(self):
        """Test generated config has correct structure."""
        config = _generate_config([])
        lines = config.split("\n")

        # Should have modules section
        assert any("modules" in line for line in lines)
        # Should have plugins section
        assert any("plugins" in line for line in lines)
        # Should have dev_mode
        assert any("dev_mode" in line for line in lines)
        # Should have settings section
        assert any("[settings]" in line for line in lines)


class TestPrintAdapterInfo:
    """Test the _print_adapter_info helper function."""

    def test_print_openai_adapter_info(self, mock_console):
        """Test printing OpenAI adapter information."""
        with patch("shutil.which", return_value="/usr/bin/uv"):
            _print_adapter_info("openai")
            assert mock_console.print.called
            # Check that calls mention OpenAI
            calls = [str(call) for call in mock_console.print.call_args_list]
            assert any("OpenAI" in str(call) for call in calls)

    def test_print_anthropic_adapter_info(self, mock_console):
        """Test printing Anthropic adapter information."""
        with patch("shutil.which", return_value=None):  # No uv available
            _print_adapter_info("anthropic")
            assert mock_console.print.called
            calls = [str(call) for call in mock_console.print.call_args_list]
            assert any("Anthropic" in str(call) for call in calls)

    def test_print_local_adapter_info(self, mock_console):
        """Test printing local adapter information."""
        _print_adapter_info("local")
        assert mock_console.print.called
        calls = [str(call) for call in mock_console.print.call_args_list]
        assert any("Local" in str(call) for call in calls)

    def test_print_mock_adapter_info(self, mock_console):
        """Test printing mock adapter information."""
        _print_adapter_info("mock")
        assert mock_console.print.called
        calls = [str(call) for call in mock_console.print.call_args_list]
        assert any("Mock" in str(call) for call in calls)

    def test_print_unknown_adapter_info(self, mock_console):
        """Test printing info for unknown adapter."""
        _print_adapter_info("unknown")
        # Should not crash, may or may not print anything

    def test_print_adapter_info_detects_uv(self, mock_console):
        """Test that adapter info uses 'uv pip install' when uv is available."""
        with patch("shutil.which", return_value="/usr/bin/uv"):
            _print_adapter_info("openai")
            calls = [str(call) for call in mock_console.print.call_args_list]
            assert any("uv pip install" in str(call) for call in calls)

    def test_print_adapter_info_uses_pip_fallback(self, mock_console):
        """Test that adapter info uses 'pip install' when uv is not available."""
        with patch("shutil.which", return_value=None):
            _print_adapter_info("openai")
            calls = [str(call) for call in mock_console.print.call_args_list]
            assert any(
                "pip install" in str(call) and "uv pip install" not in str(call) for call in calls
            )
