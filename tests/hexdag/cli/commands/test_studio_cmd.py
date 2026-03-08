"""Tests for hexdag.cli.commands.studio_cmd module."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from hexdag.cli.commands.studio_cmd import app


@pytest.fixture
def runner():
    """Fixture providing a Typer CLI test runner."""
    return CliRunner()


class TestStudioCommand:
    """Tests for studio command."""

    def test_studio_missing_dependencies(self, runner, tmp_path) -> None:
        """Studio exits with error when fastapi/uvicorn not installed."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            if name in ("fastapi", "uvicorn"):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = runner.invoke(app, [str(tmp_path)])
            assert result.exit_code == 1

    def test_studio_starts_server(self, runner, tmp_path) -> None:
        """Studio starts server when dependencies are available."""
        with (
            patch("hexdag.cli.commands.studio_cmd.console"),
            patch("hexdag.studio.server.main.run_server") as mock_server,
        ):
            result = runner.invoke(app, ["--no-browser", str(tmp_path)])
            assert result.exit_code == 0
            mock_server.assert_called_once()

    def test_studio_custom_port(self, runner, tmp_path) -> None:
        """Studio respects custom port."""
        with (
            patch("hexdag.cli.commands.studio_cmd.console"),
            patch("hexdag.studio.server.main.run_server") as mock_server,
        ):
            result = runner.invoke(app, ["--no-browser", "--port", "8080", str(tmp_path)])
            assert result.exit_code == 0
            call_args = mock_server.call_args
            assert call_args[1].get("port") == 8080 or 8080 in call_args[0]
