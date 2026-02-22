"""Tests for YAML custom tag discovery and introspection."""

import pytest
import yaml

from hexdag.kernel.pipeline_builder.tag_discovery import (
    _get_tag_syntax,
    discover_tags,
    get_known_tag_names,
    get_tag_schema,
)


class TestDiscoverTags:
    """Tests for the discover_tags function."""

    def test_discovers_py_tag(self) -> None:
        """The !py tag should be discovered."""
        tags = discover_tags()
        assert "!py" in tags
        assert tags["!py"]["name"] == "!py"
        assert "hexdag.kernel.pipeline_builder.py_tag" in tags["!py"]["module"]

    def test_discovers_include_tag(self) -> None:
        """The !include tag should be discovered."""
        tags = discover_tags()
        assert "!include" in tags
        assert tags["!include"]["name"] == "!include"
        assert "hexdag.kernel.pipeline_builder.include_tag" in tags["!include"]["module"]

    def test_tags_have_descriptions(self) -> None:
        """All tags should have descriptions."""
        tags = discover_tags()
        for tag_name, tag_info in tags.items():
            assert tag_info["description"], f"Tag {tag_name} missing description"

    def test_tags_have_documentation(self) -> None:
        """Tags should have full documentation from docstrings."""
        tags = discover_tags()
        for tag_name, tag_info in tags.items():
            assert tag_info["documentation"], f"Tag {tag_name} missing documentation"

    def test_tags_are_registered(self) -> None:
        """Discovered tags should be registered with YAML SafeLoader."""
        tags = discover_tags()
        for tag_name, tag_info in tags.items():
            assert tag_info["is_registered"], f"Tag {tag_name} not registered"
            assert tag_name in yaml.SafeLoader.yaml_constructors

    def test_tags_have_syntax_info(self) -> None:
        """Tags should have syntax information."""
        tags = discover_tags()
        for tag_name, tag_info in tags.items():
            assert tag_info["syntax"], f"Tag {tag_name} missing syntax info"

    def test_discover_tags_is_cached(self) -> None:
        """discover_tags should return cached results."""
        tags1 = discover_tags()
        tags2 = discover_tags()
        assert tags1 is tags2


class TestGetTagSchema:
    """Tests for the get_tag_schema function."""

    def test_get_py_tag_schema(self) -> None:
        """Get schema for !py tag."""
        schema = get_tag_schema("!py")
        assert schema["name"] == "!py"
        assert schema["type"] == "yaml_tag"
        assert "security_warning" in schema
        assert "input_schema" in schema
        assert schema["input_schema"]["type"] == "string"

    def test_get_include_tag_schema(self) -> None:
        """Get schema for !include tag."""
        schema = get_tag_schema("!include")
        assert schema["name"] == "!include"
        assert schema["type"] == "yaml_tag"
        assert "input_schema" in schema
        assert "oneOf" in schema["input_schema"]

    def test_tag_without_prefix(self) -> None:
        """Tags can be looked up without ! prefix."""
        schema = get_tag_schema("py")
        assert schema["name"] == "!py"

    def test_include_tag_without_prefix(self) -> None:
        """Include tag can be looked up without ! prefix."""
        schema = get_tag_schema("include")
        assert schema["name"] == "!include"

    def test_unknown_tag_raises_error(self) -> None:
        """Unknown tags should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown tag"):
            get_tag_schema("!nonexistent")

    def test_py_tag_has_security_warning(self) -> None:
        """The !py tag should have a security warning."""
        schema = get_tag_schema("!py")
        assert "security_warning" in schema
        assert "arbitrary Python code" in schema["security_warning"]

    def test_include_tag_has_no_security_warning(self) -> None:
        """The !include tag should not have a security warning."""
        schema = get_tag_schema("!include")
        assert "security_warning" not in schema

    def test_schema_has_output_info(self) -> None:
        """Schemas should include output information."""
        py_schema = get_tag_schema("!py")
        assert "output" in py_schema
        assert py_schema["output"]["type"] == "callable"

        include_schema = get_tag_schema("!include")
        assert "output" in include_schema
        assert include_schema["output"]["type"] == "any"


class TestGetKnownTagNames:
    """Tests for get_known_tag_names function."""

    def test_returns_frozenset(self) -> None:
        """Should return a frozenset."""
        names = get_known_tag_names()
        assert isinstance(names, frozenset)

    def test_contains_known_tags(self) -> None:
        """Should contain known tag names."""
        names = get_known_tag_names()
        assert "!py" in names
        assert "!include" in names

    def test_frozenset_is_immutable(self) -> None:
        """The returned set should be immutable."""
        names = get_known_tag_names()
        with pytest.raises(AttributeError):
            names.add("!new")  # type: ignore[attr-defined]


class TestGetTagSyntax:
    """Tests for the _get_tag_syntax helper."""

    def test_py_tag_syntax(self) -> None:
        """Should return syntax info for !py tag."""
        syntax = _get_tag_syntax("!py")
        assert len(syntax) > 0
        assert any("!py" in s for s in syntax)

    def test_include_tag_syntax(self) -> None:
        """Should return syntax info for !include tag."""
        syntax = _get_tag_syntax("!include")
        assert len(syntax) > 0
        assert any("!include" in s for s in syntax)

    def test_unknown_tag_returns_empty(self) -> None:
        """Unknown tags should return empty list."""
        syntax = _get_tag_syntax("!unknown")
        assert syntax == []
