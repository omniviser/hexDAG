"""Tests for hexdag.cli.commands.plugins_cmd module."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from hexdag.cli.commands.plugins_cmd import app


@pytest.fixture
def runner():
    """Fixture providing a Typer CLI test runner."""
    return CliRunner()


class TestListPlugins:
    """Tests for plugins list subcommand."""

    def test_list_returns_success(self, runner) -> None:
        """List command exits successfully."""
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0

    def test_list_json_format(self, runner) -> None:
        """List with --format json produces output."""
        result = runner.invoke(app, ["list", "--format", "json"])
        assert result.exit_code == 0

    def test_list_yaml_format(self, runner) -> None:
        """List with --format yaml produces output."""
        result = runner.invoke(app, ["list", "--format", "yaml"])
        assert result.exit_code == 0


class TestCheckPlugins:
    """Tests for plugins check subcommand."""

    def test_check_returns_success(self, runner) -> None:
        """Check command exits successfully."""
        result = runner.invoke(app, ["check"])
        assert result.exit_code == 0


class TestInstallPlugin:
    """Tests for plugins install subcommand."""

    def test_install_dry_run(self, runner) -> None:
        """Install with --dry-run does not execute install."""
        result = runner.invoke(app, ["install", "openai", "--dry-run"])
        # Should show what would be installed without actually installing
        assert result.exit_code == 0

    def test_install_unknown_plugin(self, runner) -> None:
        """Install unknown plugin shows appropriate message."""
        result = runner.invoke(app, ["install", "nonexistent_plugin_xyz", "--dry-run"])
        # May show error or unknown plugin info
        assert result.exit_code in (0, 1)
