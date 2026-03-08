"""Tests for hexdag.cli.commands.docs_cmd module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from hexdag.cli.commands.docs_cmd import app


@pytest.fixture
def runner():
    """Fixture providing a Typer CLI test runner."""
    return CliRunner()


class TestBuildDocs:
    """Tests for docs build subcommand."""

    def test_build_calls_mkdocs(self, runner, tmp_path) -> None:
        """Build command invokes mkdocs build subprocess."""
        with patch("hexdag.cli.commands.docs_cmd.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = runner.invoke(app, ["build"])
            assert result.exit_code == 0
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert "mkdocs" in args
            assert "build" in args

    def test_build_with_clean(self, runner) -> None:
        """Build passes --clean flag to mkdocs."""
        with patch("hexdag.cli.commands.docs_cmd.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = runner.invoke(app, ["build", "--clean"])
            assert result.exit_code == 0
            args = mock_run.call_args[0][0]
            assert "--clean" in args

    def test_build_with_strict(self, runner) -> None:
        """Build passes --strict flag to mkdocs."""
        with patch("hexdag.cli.commands.docs_cmd.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = runner.invoke(app, ["build", "--strict"])
            assert result.exit_code == 0
            args = mock_run.call_args[0][0]
            assert "--strict" in args

    def test_build_failure_exits_nonzero(self, runner) -> None:
        """Build exits with code 1 when mkdocs fails."""
        with patch("hexdag.cli.commands.docs_cmd.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            result = runner.invoke(app, ["build"])
            assert result.exit_code == 1


class TestNewPage:
    """Tests for docs new subcommand."""

    def test_new_page_creates_file(self, runner, tmp_path) -> None:
        """New page creates markdown file with template."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create docs directory first (required by the command)
            Path("docs").mkdir()
            result = runner.invoke(app, ["new", "guide.md", "--title", "My Guide"])
            assert result.exit_code == 0
            assert Path("docs/guide.md").exists()

    def test_new_page_creates_parent_dirs(self, runner, tmp_path) -> None:
        """New page creates parent directories."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            Path("docs").mkdir()
            result = runner.invoke(app, ["new", "deep/nested/page.md"])
            assert result.exit_code == 0
            assert Path("docs/deep/nested/page.md").exists()
