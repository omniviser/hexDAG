"""Tests for MCP server (hexdag.mcp_server)."""

import json

import pytest

# Skip all tests in this module if mcp is not installed
mcp = pytest.importorskip("mcp", reason="MCP package not installed")


class TestListTagsMCP:
    """Tests for the list_tags MCP tool."""

    def test_list_tags_returns_json(self) -> None:
        """list_tags should return valid JSON."""
        from hexdag.mcp_server import list_tags

        result = list_tags()
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_list_tags_contains_py(self) -> None:
        """list_tags should include !py tag."""
        from hexdag.mcp_server import list_tags

        result = json.loads(list_tags())
        assert "!py" in result
        assert result["!py"]["name"] == "!py"
        assert "security_warning" in result["!py"]

    def test_list_tags_contains_include(self) -> None:
        """list_tags should include !include tag."""
        from hexdag.mcp_server import list_tags

        result = json.loads(list_tags())
        assert "!include" in result
        assert result["!include"]["name"] == "!include"

    def test_list_tags_has_required_fields(self) -> None:
        """Each tag should have required fields."""
        from hexdag.mcp_server import list_tags

        result = json.loads(list_tags())
        for tag_name, tag_info in result.items():
            assert "name" in tag_info, f"Tag {tag_name} missing 'name'"
            assert "description" in tag_info, f"Tag {tag_name} missing 'description'"
            assert "module" in tag_info, f"Tag {tag_name} missing 'module'"
            assert "syntax" in tag_info, f"Tag {tag_name} missing 'syntax'"
            assert "is_registered" in tag_info, f"Tag {tag_name} missing 'is_registered'"

    def test_list_tags_py_has_security_warning(self) -> None:
        """The !py tag should have a security warning in list_tags output."""
        from hexdag.mcp_server import list_tags

        result = json.loads(list_tags())
        assert "security_warning" in result["!py"]
        assert "arbitrary Python code" in result["!py"]["security_warning"]

    def test_list_tags_include_has_no_security_warning(self) -> None:
        """The !include tag should not have a security warning."""
        from hexdag.mcp_server import list_tags

        result = json.loads(list_tags())
        assert "security_warning" not in result["!include"]


class TestGetComponentSchemaTags:
    """Tests for get_component_schema with tag type."""

    def test_get_py_tag_schema(self) -> None:
        """get_component_schema should work for !py tag."""
        from hexdag.mcp_server import get_component_schema

        result = json.loads(get_component_schema("tag", "!py"))
        assert result["name"] == "!py"
        assert result["type"] == "yaml_tag"
        assert "schema" in result
        assert "yaml_example" in result

    def test_get_include_tag_schema(self) -> None:
        """get_component_schema should work for !include tag."""
        from hexdag.mcp_server import get_component_schema

        result = json.loads(get_component_schema("tag", "!include"))
        assert result["name"] == "!include"
        assert result["type"] == "yaml_tag"

    def test_get_tag_schema_without_prefix(self) -> None:
        """Tag lookup should work without ! prefix."""
        from hexdag.mcp_server import get_component_schema

        result = json.loads(get_component_schema("tag", "py"))
        assert result["name"] == "!py"

    def test_get_include_tag_without_prefix(self) -> None:
        """Include tag lookup should work without ! prefix."""
        from hexdag.mcp_server import get_component_schema

        result = json.loads(get_component_schema("tag", "include"))
        assert result["name"] == "!include"

    def test_get_unknown_tag_returns_error(self) -> None:
        """Unknown tag should return error object."""
        from hexdag.mcp_server import get_component_schema

        result = json.loads(get_component_schema("tag", "!nonexistent"))
        assert "error" in result

    def test_tag_schema_has_yaml_example(self) -> None:
        """Tag schemas should include YAML examples."""
        from hexdag.mcp_server import get_component_schema

        for tag in ["!py", "!include"]:
            result = json.loads(get_component_schema("tag", tag))
            assert "yaml_example" in result
            assert result["yaml_example"], f"Tag {tag} missing yaml_example"

    def test_py_tag_schema_has_security_warning(self) -> None:
        """The !py tag schema should include security warning."""
        from hexdag.mcp_server import get_component_schema

        result = json.loads(get_component_schema("tag", "!py"))
        assert "security_warning" in result
        assert "arbitrary Python code" in result["security_warning"]

    def test_tag_schema_has_documentation(self) -> None:
        """Tag schemas should include full documentation."""
        from hexdag.mcp_server import get_component_schema

        result = json.loads(get_component_schema("tag", "!py"))
        assert "documentation" in result
        assert len(result["documentation"]) > 0

    def test_tag_schema_has_syntax_patterns(self) -> None:
        """Tag schemas should include syntax patterns."""
        from hexdag.mcp_server import get_component_schema

        result = json.loads(get_component_schema("tag", "!include"))
        assert "syntax" in result
        assert len(result["syntax"]) > 0

    def test_tag_schema_has_output_info(self) -> None:
        """Tag schemas should include output information."""
        from hexdag.mcp_server import get_component_schema

        py_result = json.loads(get_component_schema("tag", "!py"))
        assert "output" in py_result
        assert py_result["output"]["type"] == "callable"

        include_result = json.loads(get_component_schema("tag", "!include"))
        assert "output" in include_result
        assert include_result["output"]["type"] == "any"
