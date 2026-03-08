"""Tests for hexdag.cli.commands.validate_cmd module."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import typer
import yaml

from hexdag.cli.commands.validate_cmd import validate


class TestValidate:
    """Tests for validate command."""

    def test_validate_valid_pipeline(self, tmp_path) -> None:
        """Validate succeeds on a well-formed pipeline."""
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

        with patch("hexdag.cli.commands.validate_cmd.console"):
            try:
                validate(yaml_file=yaml_file, explain=False)
            except (SystemExit, typer.Exit) as e:
                # Exit code 0 is acceptable
                exit_code = getattr(e, "exit_code", getattr(e, "code", 0))
                assert exit_code in (0, None)

    def test_validate_nonexistent_file(self, tmp_path) -> None:
        """Validate fails on nonexistent file."""
        with (
            patch("hexdag.cli.commands.validate_cmd.console"),
            pytest.raises((SystemExit, typer.Exit, FileNotFoundError)),
        ):
            validate(yaml_file=tmp_path / "missing.yaml", explain=False)

    def test_validate_invalid_yaml(self, tmp_path) -> None:
        """Validate handles invalid YAML gracefully."""
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("not: [valid: yaml: {{")

        with patch("hexdag.cli.commands.validate_cmd.console"):
            try:
                validate(yaml_file=bad_yaml, explain=False)
            except (SystemExit, typer.Exit):
                pass  # Expected

    def test_validate_with_explain(self, tmp_path) -> None:
        """Validate with --explain produces detailed output."""
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
        yaml_file = tmp_path / "explain.yaml"
        yaml_file.write_text(yaml.dump(pipeline))

        with patch("hexdag.cli.commands.validate_cmd.console"):
            try:
                validate(yaml_file=yaml_file, explain=True)
            except (SystemExit, typer.Exit) as e:
                exit_code = getattr(e, "exit_code", getattr(e, "code", 0))
                assert exit_code in (0, None)
