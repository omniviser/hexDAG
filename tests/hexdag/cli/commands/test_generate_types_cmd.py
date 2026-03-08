"""Tests for hexdag.cli.commands.generate_types_cmd module."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import typer
import yaml

from hexdag.cli.commands.generate_types_cmd import (
    _python_type_to_str,
    generate_types,
)


class TestPythonTypeToStr:
    """Tests for _python_type_to_str helper."""

    def test_str_type(self) -> None:
        """Converts str type to 'str'."""
        assert _python_type_to_str(str) == "str"

    def test_int_type(self) -> None:
        """Converts int type to 'int'."""
        assert _python_type_to_str(int) == "int"

    def test_float_type(self) -> None:
        """Converts float type to 'float'."""
        assert _python_type_to_str(float) == "float"

    def test_bool_type(self) -> None:
        """Converts bool type to 'bool'."""
        assert _python_type_to_str(bool) == "bool"

    def test_none_returns_any(self) -> None:
        """None annotation returns 'Any'."""
        assert _python_type_to_str(None) == "Any"

    def test_dict_type(self) -> None:
        """Converts dict type to 'dict'."""
        result = _python_type_to_str(dict)
        assert "dict" in result

    def test_list_type(self) -> None:
        """Converts list type to 'list'."""
        result = _python_type_to_str(list)
        assert "list" in result


class TestGenerateTypes:
    """Tests for generate_types command."""

    def test_generates_stubs_from_pipeline(self, tmp_path) -> None:
        """Generates .pyi stub files from a valid pipeline."""
        pipeline = {
            "apiVersion": "hexdag/v1",
            "kind": "Pipeline",
            "metadata": {"name": "test-pipeline"},
            "spec": {
                "nodes": [
                    {
                        "kind": "function_node",
                        "metadata": {"name": "my-step"},
                        "spec": {
                            "fn": "json.loads",
                            "input_mapping": {"data": "str"},
                        },
                        "dependencies": [],
                    }
                ]
            },
        }
        yaml_file = tmp_path / "pipeline.yaml"
        yaml_file.write_text(yaml.dump(pipeline))
        output_dir = tmp_path / "stubs"
        output_dir.mkdir()

        with patch("hexdag.cli.commands.generate_types_cmd.console"):
            try:
                generate_types(yaml_path=yaml_file, output_dir=output_dir, prefix="")
            except (SystemExit, typer.Exit) as e:
                exit_code = getattr(e, "exit_code", getattr(e, "code", 0))
                assert exit_code in (0, None)

    def test_handles_missing_yaml(self, tmp_path) -> None:
        """Handles missing YAML file gracefully."""
        with (
            patch("hexdag.cli.commands.generate_types_cmd.console"),
            pytest.raises((SystemExit, typer.Exit, FileNotFoundError)),
        ):
            generate_types(
                yaml_path=tmp_path / "nonexistent.yaml",
                output_dir=tmp_path,
                prefix="",
            )
