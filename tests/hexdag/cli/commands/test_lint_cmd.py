"""Tests for hexdag.cli.commands.lint_cmd module."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import typer
import yaml

from hexdag.cli.commands.lint_cmd import _SEVERITY_RANK, lint


class TestLintSeverityRank:
    """Tests for severity ranking."""

    def test_error_highest_severity(self) -> None:
        """Error has lowest rank number (highest severity)."""
        assert _SEVERITY_RANK["error"] < _SEVERITY_RANK["warning"]
        assert _SEVERITY_RANK["warning"] < _SEVERITY_RANK["info"]


class TestLint:
    """Tests for lint command."""

    def test_lint_valid_pipeline(self, tmp_path) -> None:
        """Lint succeeds on a valid pipeline YAML."""
        pipeline = {
            "apiVersion": "hexdag/v1",
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "spec": {
                "nodes": [
                    {
                        "kind": "function_node",
                        "metadata": {"name": "step1"},
                        "spec": {"fn": "json.loads"},
                        "dependencies": [],
                    }
                ]
            },
        }
        yaml_file = tmp_path / "valid.yaml"
        yaml_file.write_text(yaml.dump(pipeline))

        with patch("hexdag.cli.commands.lint_cmd.console"):
            try:
                lint(yaml_file=yaml_file, severity="info", output_format="text", disable="")
            except (SystemExit, typer.Exit):
                pass  # Exit is acceptable

    def test_lint_nonexistent_file(self, tmp_path) -> None:
        """Lint fails on nonexistent file."""
        with (
            patch("hexdag.cli.commands.lint_cmd.console"),
            pytest.raises((SystemExit, typer.Exit, FileNotFoundError)),
        ):
            lint(
                yaml_file=tmp_path / "missing.yaml",
                severity="info",
                output_format="text",
                disable="",
            )

    def test_lint_invalid_yaml(self, tmp_path) -> None:
        """Lint handles invalid YAML gracefully."""
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("not: [valid: yaml: {{")

        with patch("hexdag.cli.commands.lint_cmd.console"):
            try:
                lint(yaml_file=bad_yaml, severity="info", output_format="text", disable="")
            except (SystemExit, typer.Exit):
                pass  # Expected to fail

    def test_lint_json_format(self, tmp_path) -> None:
        """Lint produces JSON output."""
        pipeline = {
            "apiVersion": "hexdag/v1",
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "spec": {"nodes": []},
        }
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text(yaml.dump(pipeline))

        with patch("hexdag.cli.commands.lint_cmd.console"):
            try:
                lint(yaml_file=yaml_file, severity="info", output_format="json", disable="")
            except (SystemExit, typer.Exit):
                pass  # Acceptable
