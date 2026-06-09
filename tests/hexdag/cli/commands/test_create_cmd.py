"""Tests for hexdag.cli.commands.create_cmd module."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from hexdag.cli.commands.create_cmd import app


@pytest.fixture
def runner():
    """Fixture providing a Typer CLI test runner."""
    return CliRunner()


@pytest.fixture
def _chdir(tmp_path, monkeypatch):
    """Change working directory to tmp_path for the test."""
    monkeypatch.chdir(tmp_path)


@pytest.mark.usefixtures("_chdir")
class TestCreatePipeline:
    """Tests for create pipeline command."""

    def test_creates_minimal_template(self, runner) -> None:
        """Create command generates a minimal pipeline YAML."""
        result = runner.invoke(app, ["--name", "test_pipe", "--template", "minimal"])
        assert result.exit_code == 0
        assert Path("test_pipe.yaml").exists()

    def test_creates_with_custom_output(self, runner) -> None:
        """Create command respects custom output path."""
        result = runner.invoke(app, ["--name", "test", "--output", "custom.yaml"])
        assert result.exit_code == 0
        assert Path("custom.yaml").exists()

    def test_creates_example_template(self, runner) -> None:
        """Create command generates example template."""
        result = runner.invoke(app, ["--name", "ex", "--template", "example"])
        assert result.exit_code == 0
        assert Path("ex.yaml").exists()

    def test_creates_full_template(self, runner) -> None:
        """Create command generates full template."""
        result = runner.invoke(app, ["--name", "full", "--template", "full"])
        assert result.exit_code == 0
        assert Path("full.yaml").exists()

    def test_generated_yaml_is_valid(self, runner) -> None:
        """Generated YAML is parseable."""
        import yaml

        runner.invoke(app, ["--name", "valid_test", "--template", "minimal"])
        content = Path("valid_test.yaml").read_text()
        parsed = yaml.safe_load(content)
        assert isinstance(parsed, dict)
        assert parsed["apiVersion"] == "hexdag/v1"

    def test_abort_on_existing_file(self, runner) -> None:
        """Aborts when file exists and user declines overwrite."""
        Path("existing.yaml").write_text("old content")
        runner.invoke(
            app,
            ["--name", "existing"],
            input="n\n",
        )
        # File should not be overwritten
        assert Path("existing.yaml").read_text() == "old content"
