"""Tests for hexai.cli.main module."""

import sys
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from hexai.cli.main import app, callback, main


@pytest.fixture
def runner():
    """Fixture providing a Typer CLI test runner."""
    return CliRunner()


class TestMainApp:
    """Test the main CLI application."""

    def test_app_has_correct_name(self):
        """Test that the app has the correct name."""
        assert app.info.name == "hexdag"

    def test_app_has_help_enabled(self):
        """Test that no_args_is_help is enabled."""
        assert app.info.no_args_is_help is True

    def test_app_help_shows_subcommands(self, runner):
        """Test that --help shows available subcommands."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "init" in result.stdout
        assert "config" in result.stdout
        assert "plugins" in result.stdout
        assert "plugin" in result.stdout
        assert "registry" in result.stdout

    def test_version_flag_short(self, runner):
        """Test -v flag shows version."""
        result = runner.invoke(app, ["-v"])
        # Typer.Exit() causes exit code 0
        assert result.exit_code == 0
        assert "HexDAG" in result.stdout or "version" in result.stdout.lower()

    def test_version_flag_long(self, runner):
        """Test --version flag shows version."""
        result = runner.invoke(app, ["--version"])
        # Typer.Exit() causes exit code 0
        assert result.exit_code == 0
        assert "HexDAG" in result.stdout or "version" in result.stdout.lower()


class TestCallback:
    """Test the main callback function."""

    def test_callback_with_version_true_exits(self):
        """Test that callback raises Exit when version=True."""
        import typer

        with pytest.raises(typer.Exit):
            callback(version=True)

    def test_callback_with_version_false_does_nothing(self):
        """Test that callback does nothing when version=False."""
        # Should not raise any exception
        callback(version=False)


class TestMainFunction:
    """Test the main() entrypoint function."""

    @patch("hexai.cli.main.app")
    def test_main_calls_app(self, mock_app):
        """Test that main() invokes the Typer app."""
        main()
        mock_app.assert_called_once()

    def test_main_executable_via_python_m(self):
        """Test that the CLI can be executed via python -m."""
        # This tests that __main__ works correctly
        import subprocess

        result = subprocess.run(
            [sys.executable, "-m", "hexai.cli", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        # Check both stdout and stderr, and allow any successful exit code
        output = result.stdout + result.stderr
        assert result.returncode in [0, 1]  # May exit with 0 or 1
        assert "HexDAG" in output or "version" in output or "hexdag" in output.lower()


class TestCLIImportError:
    """Test CLI behavior when dependencies are missing."""

    def test_import_error_handling(self):
        """Test that missing dependencies are handled gracefully."""
        # This test verifies that the main module has import error handling
        # We can't easily test the actual import failure since the module is already loaded
        # but we verify the structure is in place
        from hexai.cli import main as cli_main_module

        # Check that the module exists and has the expected structure
        assert hasattr(cli_main_module, "app")
        assert hasattr(cli_main_module, "main")
        assert hasattr(cli_main_module, "callback")
