"""Tests for the schema conversion module.

This module tests utilities for converting YAML schema strings to Python types.
"""

from __future__ import annotations

from typing import Any

import pytest

from hexdag.core.utils.schema_conversion import (
    VALID_TYPE_NAMES,
    convert_yaml_schema,
    is_valid_type_name,
    normalize_schema,
)


class TestValidTypeNames:
    """Tests for VALID_TYPE_NAMES mapping."""

    def test_contains_basic_types(self) -> None:
        """Test that basic types are in the mapping."""
        assert "str" in VALID_TYPE_NAMES
        assert "int" in VALID_TYPE_NAMES
        assert "float" in VALID_TYPE_NAMES
        assert "bool" in VALID_TYPE_NAMES

    def test_contains_container_types(self) -> None:
        """Test that container types are in the mapping."""
        assert "list" in VALID_TYPE_NAMES
        assert "dict" in VALID_TYPE_NAMES

    def test_contains_any_type(self) -> None:
        """Test that Any type is in the mapping."""
        assert "Any" in VALID_TYPE_NAMES
        assert VALID_TYPE_NAMES["Any"] is Any

    def test_maps_to_correct_types(self) -> None:
        """Test that mappings are correct."""
        assert VALID_TYPE_NAMES["str"] is str
        assert VALID_TYPE_NAMES["int"] is int
        assert VALID_TYPE_NAMES["float"] is float
        assert VALID_TYPE_NAMES["bool"] is bool
        assert VALID_TYPE_NAMES["list"] is list
        assert VALID_TYPE_NAMES["dict"] is dict


class TestNormalizeSchemaDefault:
    """Tests for normalize_schema default behavior."""

    def test_passthrough_type(self) -> None:
        """Test that types pass through unchanged."""
        assert normalize_schema(str) is str
        assert normalize_schema(int) is int
        assert normalize_schema(list) is list

    def test_passthrough_none(self) -> None:
        """Test that None passes through."""
        assert normalize_schema(None) is None

    def test_passthrough_string(self) -> None:
        """Test that plain strings pass through (not in dict)."""
        assert normalize_schema("some_string") == "some_string"

    def test_passthrough_list(self) -> None:
        """Test that lists pass through."""
        result = normalize_schema([1, 2, 3])
        assert result == [1, 2, 3]


class TestNormalizeSchemaDict:
    """Tests for normalize_schema dict handling."""

    def test_convert_string_type_names(self) -> None:
        """Test converting string type names to types."""
        schema = {"name": "str", "age": "int"}
        result = normalize_schema(schema)
        assert result == {"name": str, "age": int}

    def test_convert_all_valid_types(self) -> None:
        """Test converting all valid type names."""
        schema = {
            "string_field": "str",
            "int_field": "int",
            "float_field": "float",
            "bool_field": "bool",
            "list_field": "list",
            "dict_field": "dict",
            "any_field": "Any",
        }
        result = normalize_schema(schema)
        assert result["string_field"] is str
        assert result["int_field"] is int
        assert result["float_field"] is float
        assert result["bool_field"] is bool
        assert result["list_field"] is list
        assert result["dict_field"] is dict
        assert result["any_field"] is Any

    def test_passthrough_type_objects(self) -> None:
        """Test that type objects in dict pass through."""
        schema = {"name": str, "count": int}
        result = normalize_schema(schema)
        assert result == {"name": str, "count": int}

    def test_mixed_strings_and_types(self) -> None:
        """Test dict with mix of string names and type objects."""
        schema = {"name": "str", "count": int, "active": "bool"}
        result = normalize_schema(schema)
        assert result == {"name": str, "count": int, "active": bool}

    def test_nested_schema(self) -> None:
        """Test normalizing nested schema dicts."""
        schema = {
            "user": {
                "name": "str",
                "age": "int",
            }
        }
        result = normalize_schema(schema)
        assert result == {"user": {"name": str, "age": int}}

    def test_deeply_nested_schema(self) -> None:
        """Test deeply nested schema dicts."""
        schema = {
            "level1": {
                "level2": {
                    "level3": {
                        "value": "str",
                    }
                }
            }
        }
        result = normalize_schema(schema)
        assert result["level1"]["level2"]["level3"]["value"] is str

    def test_empty_dict(self) -> None:
        """Test normalizing empty dict."""
        result = normalize_schema({})
        assert result == {}

    def test_invalid_type_name_raises_error(self) -> None:
        """Test that invalid type name raises ValueError."""
        schema = {"field": "invalid_type"}
        with pytest.raises(ValueError) as exc_info:
            normalize_schema(schema)
        assert "Invalid type 'invalid_type'" in str(exc_info.value)
        assert "field" in str(exc_info.value)
        assert "Supported types" in str(exc_info.value)

    def test_invalid_value_type_raises_error(self) -> None:
        """Test that invalid value type raises ValueError."""
        schema = {"field": 123}  # Not a string, type, or dict
        with pytest.raises(ValueError) as exc_info:
            normalize_schema(schema)
        assert "invalid value" in str(exc_info.value)
        assert "field" in str(exc_info.value)


class TestNullableTypes:
    """Tests for nullable type syntax (str?, int?, etc.)."""

    def test_nullable_types_accepted(self) -> None:
        """normalize_schema accepts ?-suffixed types and passes them through."""
        result = normalize_schema({"name": "str?", "count": "int?"})
        assert result == {"name": "str?", "count": "int?"}

    def test_all_nullable_variants_accepted(self) -> None:
        for t in ("str?", "int?", "float?", "bool?", "list?", "dict?"):
            result = normalize_schema({"field": t})
            assert result == {"field": t}

    def test_mixed_required_and_nullable(self) -> None:
        schema = {"rate": "float", "origin": "str?", "mc_number": "str?"}
        result = normalize_schema(schema)
        assert result["rate"] is float
        assert result["origin"] == "str?"
        assert result["mc_number"] == "str?"

    def test_invalid_nullable_raises_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid type 'invalid\\?'"):
            normalize_schema({"field": "invalid?"})


class TestIsValidTypeName:
    """Tests for is_valid_type_name helper."""

    def test_base_types(self) -> None:
        for t in ("str", "int", "float", "bool", "list", "dict", "Any"):
            assert is_valid_type_name(t)

    def test_nullable_types(self) -> None:
        for t in ("str?", "int?", "float?", "bool?", "list?", "dict?"):
            assert is_valid_type_name(t)

    def test_invalid_types(self) -> None:
        assert not is_valid_type_name("invalid")
        assert not is_valid_type_name("invalid?")
        assert not is_valid_type_name("Any?")
        assert not is_valid_type_name("")


class TestConvertYamlSchema:
    """Tests for convert_yaml_schema backward compatibility alias."""

    def test_is_alias_for_normalize_schema(self) -> None:
        """Test that convert_yaml_schema is the same as normalize_schema."""
        assert convert_yaml_schema is normalize_schema

    def test_works_like_normalize_schema(self) -> None:
        """Test that it works the same way."""
        schema = {"name": "str", "value": "int"}
        result = convert_yaml_schema(schema)
        assert result == {"name": str, "value": int}
