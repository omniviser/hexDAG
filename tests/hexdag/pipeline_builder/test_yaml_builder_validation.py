"""Tests for YAML builder validation and error handling.

Tests the robust validation added to yaml_builder.py to ensure:
1. Invalid type names are rejected with clear errors
2. Missing required fields are caught early
3. Error messages provide helpful context
"""

import pytest

from hexdag.core.pipeline_builder.yaml_builder import (
    YamlPipelineBuilder,
    YamlPipelineBuilderError,
    _convert_schema_types,
)


class TestSchemaTypeValidation:
    """Tests for _convert_schema_types validation."""

    def test_valid_types(self):
        """Test that all valid type names are accepted."""
        schema = {
            "name": "str",
            "age": "int",
            "score": "float",
            "active": "bool",
            "items": "list",
            "mapping": "dict",
            "anything": "Any",
        }

        result = _convert_schema_types(schema)

        assert result["name"] is str
        assert result["age"] is int
        assert result["score"] is float
        assert result["active"] is bool
        assert result["items"] is list
        assert result["mapping"] is dict
        # Note: Any is typing.Any, not a builtin type

    def test_invalid_type_name_raises_error(self):
        """Test that invalid type names raise clear errors."""
        schema = {"name": "stringg"}  # Typo

        with pytest.raises(YamlPipelineBuilderError) as exc_info:
            _convert_schema_types(schema)

        error_msg = str(exc_info.value)
        assert "Invalid type 'stringg'" in error_msg
        assert "field 'name'" in error_msg
        assert "Supported types:" in error_msg
        assert "str" in error_msg  # Shows valid options

    def test_multiple_invalid_types(self):
        """Test that first invalid type is caught."""
        schema = {
            "field1": "string",  # Invalid
            "field2": "integer",  # Invalid
        }

        with pytest.raises(YamlPipelineBuilderError) as exc_info:
            _convert_schema_types(schema)

        # Should catch the first invalid type
        error_msg = str(exc_info.value)
        assert "Invalid type" in error_msg

    def test_mixed_valid_invalid_types(self):
        """Test schema with both valid and invalid types."""
        schema = {
            "valid_field": "str",
            "invalid_field": "text",  # Invalid
        }

        with pytest.raises(YamlPipelineBuilderError) as exc_info:
            _convert_schema_types(schema)

        error_msg = str(exc_info.value)
        assert "'text'" in error_msg or "'invalid_field'" in error_msg

    def test_already_type_objects(self):
        """Test that existing type objects are preserved."""
        schema = {
            "existing": str,  # Already a type object
            "new": "int",  # String to convert
        }

        result = _convert_schema_types(schema)

        assert result["existing"] is str
        assert result["new"] is int

    def test_invalid_value_type(self):
        """Test that non-string, non-type values are rejected."""
        schema = {
            "invalid": 123,  # Neither string nor type
        }

        with pytest.raises(YamlPipelineBuilderError) as exc_info:
            _convert_schema_types(schema)

        error_msg = str(exc_info.value)
        assert "invalid value" in error_msg.lower()
        assert "123" in error_msg

    def test_non_dict_schema_returns_unchanged(self):
        """Test that non-dict schemas are returned unchanged."""
        assert _convert_schema_types(None) is None
        assert _convert_schema_types("not_a_dict") == "not_a_dict"
        assert _convert_schema_types([1, 2, 3]) == [1, 2, 3]

    def test_empty_schema(self):
        """Test that empty schema dict is handled."""
        result = _convert_schema_types({})
        assert result == {}


class TestNodeConfigValidation:
    """Tests for node configuration validation."""

    def test_missing_kind_field(self):
        """Test that missing 'kind' field raises clear error."""
        builder = YamlPipelineBuilder()

        yaml_content = """
        apiVersion: hexdag.omniviser.io/v1alpha1
        kind: Pipeline
        metadata:
          name: test-pipeline
        spec:
          nodes:
            - metadata:
                name: test_node
              spec:
                prompt_template: "Test"
        """

        with pytest.raises(YamlPipelineBuilderError) as exc_info:
            builder.build_from_yaml_string(yaml_content)

        error_msg = str(exc_info.value)
        # YamlValidator catches this first
        assert "missing 'kind'" in error_msg.lower() or "kind" in error_msg.lower()

    def test_missing_metadata_section(self):
        """Test that missing 'metadata' section raises error."""
        builder = YamlPipelineBuilder()

        yaml_content = """
        apiVersion: hexdag.omniviser.io/v1alpha1
        kind: Pipeline
        metadata:
          name: test-pipeline
        spec:
          nodes:
            - kind: llm_node
              spec:
                prompt_template: "Test"
        """

        with pytest.raises(YamlPipelineBuilderError) as exc_info:
            builder.build_from_yaml_string(yaml_content)

        error_msg = str(exc_info.value)
        # YamlValidator catches this
        assert "metadata" in error_msg.lower()

    def test_missing_node_name(self):
        """Test that missing metadata.name raises error."""
        builder = YamlPipelineBuilder()

        yaml_content = """
        apiVersion: hexdag.omniviser.io/v1alpha1
        kind: Pipeline
        metadata:
          name: test-pipeline
        spec:
          nodes:
            - kind: llm_node
              metadata:
                description: "Missing name"
              spec:
                prompt_template: "Test"
        """

        with pytest.raises(YamlPipelineBuilderError) as exc_info:
            builder.build_from_yaml_string(yaml_content)

        error_msg = str(exc_info.value)
        # YamlValidator catches this
        assert "metadata.name" in error_msg.lower() or "name" in error_msg.lower()

    def test_invalid_output_schema_in_parser_node(self):
        """Test that invalid output_schema in parser node includes node name in error."""
        builder = YamlPipelineBuilder()

        yaml_content = """
        apiVersion: hexdag.omniviser.io/v1alpha1
        kind: Pipeline
        metadata:
          name: test-pipeline
        spec:
          nodes:
            - kind: parser_node
              metadata:
                name: my_parser
              spec:
                output_schema:
                  result: "stringgg"
        """

        with pytest.raises(YamlPipelineBuilderError) as exc_info:
            builder.build_from_yaml_string(yaml_content)

        error_msg = str(exc_info.value)
        assert "my_parser" in error_msg
        assert "output_schema" in error_msg.lower()
        assert "stringgg" in error_msg


class TestMacroValidation:
    """Tests for macro invocation validation."""

    def test_missing_macro_metadata(self):
        """Test that macro_invocation without metadata raises error."""
        builder = YamlPipelineBuilder()

        yaml_content = """
        apiVersion: hexdag.omniviser.io/v1alpha1
        kind: Pipeline
        metadata:
          name: test-pipeline
        spec:
          nodes:
            - kind: macro_invocation
              spec:
                macro: core:test_macro
        """

        with pytest.raises(YamlPipelineBuilderError) as exc_info:
            builder.build_from_yaml_string(yaml_content)

        error_msg = str(exc_info.value)
        # YamlValidator catches this
        assert "metadata" in error_msg.lower()

    def test_missing_macro_name(self):
        """Test that macro_invocation without name raises error."""
        builder = YamlPipelineBuilder()

        yaml_content = """
        apiVersion: hexdag.omniviser.io/v1alpha1
        kind: Pipeline
        metadata:
          name: test-pipeline
        spec:
          nodes:
            - kind: macro_invocation
              metadata:
                description: "Missing name"
              spec:
                macro: core:test_macro
        """

        with pytest.raises(YamlPipelineBuilderError) as exc_info:
            builder.build_from_yaml_string(yaml_content)

        error_msg = str(exc_info.value)
        assert "metadata.name" in error_msg.lower()

    def test_missing_macro_spec(self):
        """Test that macro_invocation without spec raises error."""
        builder = YamlPipelineBuilder()

        yaml_content = """
        apiVersion: hexdag.omniviser.io/v1alpha1
        kind: Pipeline
        metadata:
          name: test-pipeline
        spec:
          nodes:
            - kind: macro_invocation
              metadata:
                name: my_macro
        """

        with pytest.raises(YamlPipelineBuilderError) as exc_info:
            builder.build_from_yaml_string(yaml_content)

        error_msg = str(exc_info.value)
        assert "my_macro" in error_msg
        # YamlValidator or our code catches this
        assert "spec" in error_msg.lower() or "macro" in error_msg.lower()

    def test_missing_macro_reference(self):
        """Test that macro_invocation without macro field raises error."""
        builder = YamlPipelineBuilder()

        yaml_content = """
        apiVersion: hexdag.omniviser.io/v1alpha1
        kind: Pipeline
        metadata:
          name: test-pipeline
        spec:
          nodes:
            - kind: macro_invocation
              metadata:
                name: my_macro
              spec:
                inputs: {}
        """

        with pytest.raises(YamlPipelineBuilderError) as exc_info:
            builder.build_from_yaml_string(yaml_content)

        error_msg = str(exc_info.value)
        assert "my_macro" in error_msg
        assert "spec.macro" in error_msg.lower()

    def test_invalid_output_schema_in_macro(self):
        """Test that invalid output_schema in macro config includes macro name."""
        builder = YamlPipelineBuilder()

        yaml_content = """
        apiVersion: hexdag.omniviser.io/v1alpha1
        kind: Pipeline
        metadata:
          name: test-pipeline
        spec:
          nodes:
            - kind: macro_invocation
              metadata:
                name: my_macro
              spec:
                macro: core:nonexistent
                config:
                  output_schema:
                    result: "invalid_type"
        """

        with pytest.raises(YamlPipelineBuilderError) as exc_info:
            builder.build_from_yaml_string(yaml_content)

        error_msg = str(exc_info.value)
        # Should mention either the schema error or that macro doesn't exist
        assert "my_macro" in error_msg or "nonexistent" in error_msg


class TestErrorMessages:
    """Tests for error message quality and helpfulness."""

    def test_type_error_lists_valid_types(self):
        """Test that type errors list all valid type options."""
        schema = {"field": "unknown_type"}

        with pytest.raises(YamlPipelineBuilderError) as exc_info:
            _convert_schema_types(schema)

        error_msg = str(exc_info.value)
        # Should list all valid types
        for type_name in ["str", "int", "float", "bool", "list", "dict", "Any"]:
            assert type_name in error_msg

    def test_error_includes_field_name(self):
        """Test that errors identify which field is problematic."""
        schema = {"good_field": "str", "bad_field": "wrong"}

        with pytest.raises(YamlPipelineBuilderError) as exc_info:
            _convert_schema_types(schema)

        error_msg = str(exc_info.value)
        assert "bad_field" in error_msg

    def test_error_includes_invalid_value(self):
        """Test that errors show what invalid value was provided."""
        schema = {"field": "typo"}

        with pytest.raises(YamlPipelineBuilderError) as exc_info:
            _convert_schema_types(schema)

        error_msg = str(exc_info.value)
        assert "typo" in error_msg
