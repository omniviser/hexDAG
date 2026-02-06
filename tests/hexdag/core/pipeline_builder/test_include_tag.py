"""Tests for !include YAML tag handler."""

from pathlib import Path
from textwrap import dedent

import pytest
import yaml

from hexdag.core.pipeline_builder.include_tag import (
    IncludeTagError,
    _resolve_include_path,
    _substitute_vars,
    get_include_base_path,
    register_include_tag,
    set_include_base_path,
)


class TestIncludeTagRegistration:
    """Tests for !include tag registration."""

    def test_tag_is_registered(self) -> None:
        """!include tag should be registered with SafeLoader."""
        register_include_tag()
        assert "!include" in yaml.SafeLoader.yaml_constructors


class TestIncludeBasePath:
    """Tests for base path management."""

    def test_set_and_get_base_path(self, tmp_path: Path) -> None:
        """set_include_base_path and get_include_base_path work correctly."""
        set_include_base_path(tmp_path)
        assert get_include_base_path() == tmp_path

        # Reset
        set_include_base_path(None)
        assert get_include_base_path() == Path.cwd()

    def test_default_base_path_is_cwd(self) -> None:
        """Default base path is current working directory."""
        set_include_base_path(None)
        assert get_include_base_path() == Path.cwd()


class TestResolveIncludePath:
    """Tests for path resolution."""

    def test_relative_path(self, tmp_path: Path) -> None:
        """Relative paths resolve relative to base."""
        result = _resolve_include_path("./nodes.yaml", tmp_path)
        assert result == (tmp_path / "nodes.yaml").resolve()

    def test_nested_relative_path(self, tmp_path: Path) -> None:
        """Nested relative paths resolve correctly."""
        result = _resolve_include_path("shared/common.yaml", tmp_path)
        assert result == (tmp_path / "shared" / "common.yaml").resolve()

    def test_absolute_path(self, tmp_path: Path) -> None:
        """Absolute paths are used as-is."""
        abs_path = tmp_path / "absolute.yaml"
        result = _resolve_include_path(str(abs_path), Path("/other"))
        assert result == abs_path


class TestSubstituteVars:
    """Tests for variable substitution."""

    def test_string_substitution(self) -> None:
        """String variables are substituted."""
        content = "name: {{node_name}}"
        result = _substitute_vars(content, {"node_name": "my_node"})
        assert result == "name: my_node"

    def test_integer_substitution(self) -> None:
        """Integer variables are substituted."""
        content = "timeout: {{timeout}}"
        result = _substitute_vars(content, {"timeout": 30})
        assert result == "timeout: 30"

    def test_boolean_substitution(self) -> None:
        """Boolean variables are substituted as YAML booleans."""
        content = "enabled: {{flag}}"
        result = _substitute_vars(content, {"flag": True})
        assert result == "enabled: true"

        result = _substitute_vars(content, {"flag": False})
        assert result == "enabled: false"

    def test_null_substitution(self) -> None:
        """None is substituted as YAML null."""
        content = "value: {{val}}"
        result = _substitute_vars(content, {"val": None})
        assert result == "value: null"

    def test_multiple_substitutions(self) -> None:
        """Multiple variables in same content."""
        content = "name: {{name}}\ntimeout: {{timeout}}"
        result = _substitute_vars(content, {"name": "test", "timeout": 60})
        assert result == "name: test\ntimeout: 60"


class TestIncludeConstructor:
    """Tests for the !include constructor."""

    def test_include_simple_file(self, tmp_path: Path) -> None:
        """!include loads content from external file."""
        # Create include file
        include_file = tmp_path / "nodes.yaml"
        include_file.write_text(
            dedent("""
            - name: node1
              value: 10
            - name: node2
              value: 20
        """).strip()
        )

        # Set base path and parse
        set_include_base_path(tmp_path)
        try:
            result = yaml.safe_load("items: !include ./nodes.yaml")
            assert result["items"] == [
                {"name": "node1", "value": 10},
                {"name": "node2", "value": 20},
            ]
        finally:
            set_include_base_path(None)

    def test_include_with_vars(self, tmp_path: Path) -> None:
        """!include with vars substitutes variables."""
        # Create include file with placeholders
        # Note: Placeholders must be quoted to avoid YAML parsing issues with {{ }}
        include_file = tmp_path / "template.yaml"
        include_file.write_text(
            dedent("""
            name: "{{node_name}}"
            timeout: "{{timeout}}"
        """).strip()
        )

        set_include_base_path(tmp_path)
        try:
            yaml_content = dedent("""
                node: !include
                  path: ./template.yaml
                  vars:
                    node_name: custom_node
                    timeout: 45
            """).strip()
            result = yaml.safe_load(yaml_content)
            assert result["node"]["name"] == "custom_node"
            # Note: timeout is a string after substitution because template had quotes
            assert result["node"]["timeout"] == "45"
        finally:
            set_include_base_path(None)

    def test_include_nested(self, tmp_path: Path) -> None:
        """!include works with nested includes."""
        # Create nested structure
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()

        # Inner include
        inner_file = shared_dir / "inner.yaml"
        inner_file.write_text("inner_value: 42")

        # Outer include that references inner
        outer_file = tmp_path / "outer.yaml"
        outer_file.write_text(
            dedent("""
            outer_key: outer
            nested: !include ./shared/inner.yaml
        """).strip()
        )

        set_include_base_path(tmp_path)
        try:
            result = yaml.safe_load("data: !include ./outer.yaml")
            assert result["data"]["outer_key"] == "outer"
            assert result["data"]["nested"]["inner_value"] == 42
        finally:
            set_include_base_path(None)

    def test_include_file_not_found(self, tmp_path: Path) -> None:
        """!include raises error for missing file."""
        set_include_base_path(tmp_path)
        try:
            with pytest.raises(IncludeTagError, match="not found"):
                yaml.safe_load("data: !include ./nonexistent.yaml")
        finally:
            set_include_base_path(None)

    def test_include_invalid_yaml(self, tmp_path: Path) -> None:
        """!include raises error for invalid YAML."""
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("invalid: yaml: content: [")

        set_include_base_path(tmp_path)
        try:
            with pytest.raises(IncludeTagError, match="Failed to parse"):
                yaml.safe_load("data: !include ./bad.yaml")
        finally:
            set_include_base_path(None)

    def test_include_in_list(self, tmp_path: Path) -> None:
        """!include works within a list."""
        # Create fragment
        fragment_file = tmp_path / "fragment.yaml"
        fragment_file.write_text(
            dedent("""
            - kind: function_node
              metadata:
                name: from_fragment
              spec:
                fn: "myapp.process"
        """).strip()
        )

        set_include_base_path(tmp_path)
        try:
            yaml_content = dedent("""
                nodes:
                  - kind: expression_node
                    metadata:
                      name: start
                    spec:
                      expressions:
                        ready: "true"
                  - !include ./fragment.yaml
                  - kind: llm_node
                    metadata:
                      name: end
                    spec:
                      prompt_template: "Done"
            """).strip()
            result = yaml.safe_load(yaml_content)

            # The include inserts a list, so we have: [node, [included_nodes], node]
            assert len(result["nodes"]) == 3
            assert result["nodes"][0]["metadata"]["name"] == "start"
            # Included fragment is a list
            assert isinstance(result["nodes"][1], list)
            assert result["nodes"][1][0]["metadata"]["name"] == "from_fragment"
            assert result["nodes"][2]["metadata"]["name"] == "end"
        finally:
            set_include_base_path(None)


class TestIncludeTagPipelineIntegration:
    """Integration tests for !include in pipeline context."""

    def test_include_shared_nodes(self, tmp_path: Path) -> None:
        """!include can be used to share node definitions."""
        # Create shared validation nodes
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()

        validation_nodes = shared_dir / "validation.yaml"
        validation_nodes.write_text(
            dedent("""
            kind: function_node
            metadata:
              name: validate_input
            spec:
              fn: "myapp.validate"
        """).strip()
        )

        set_include_base_path(tmp_path)
        try:
            pipeline_yaml = dedent("""
                apiVersion: hexdag/v1
                kind: Pipeline
                metadata:
                  name: test-pipeline
                spec:
                  nodes:
                    - !include ./shared/validation.yaml
                    - kind: expression_node
                      metadata:
                        name: process
                      spec:
                        expressions:
                          result: "validate_input.success"
                      dependencies:
                        - validate_input
            """).strip()

            result = yaml.safe_load(pipeline_yaml)
            assert result["spec"]["nodes"][0]["metadata"]["name"] == "validate_input"
            assert result["spec"]["nodes"][1]["metadata"]["name"] == "process"
        finally:
            set_include_base_path(None)
