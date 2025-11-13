"""Test that MCP server properly handles enum serialization in YAML."""

from enum import Enum

import pytest
import yaml


# Import the helper function
from hexdag.mcp_server import _normalize_for_yaml


class TestEnumNormalization:
    """Test enum normalization for YAML serialization."""

    def test_simple_enum_normalization(self):
        """Test that enums are converted to their string values."""

        class Color(str, Enum):
            RED = "red"
            BLUE = "blue"

        input_data = {"color": Color.RED}
        normalized = _normalize_for_yaml(input_data)

        assert normalized == {"color": "red"}
        assert isinstance(normalized["color"], str)

    def test_nested_enum_normalization(self):
        """Test enum normalization in nested structures."""

        class Format(str, Enum):
            JSON = "json"
            MIXED = "mixed"

        input_data = {
            "config": {"tool_format": Format.MIXED, "output_format": Format.JSON},
            "nested": [{"format": Format.MIXED}],
        }

        normalized = _normalize_for_yaml(input_data)

        assert normalized["config"]["tool_format"] == "mixed"
        assert normalized["config"]["output_format"] == "json"
        assert normalized["nested"][0]["format"] == "mixed"

    def test_list_with_enums(self):
        """Test enum normalization in lists."""

        class Status(str, Enum):
            ACTIVE = "active"
            INACTIVE = "inactive"

        input_data = {"statuses": [Status.ACTIVE, Status.INACTIVE, "other"]}

        normalized = _normalize_for_yaml(input_data)

        assert normalized["statuses"] == ["active", "inactive", "other"]

    def test_yaml_round_trip(self):
        """Test that normalized data can be serialized and deserialized."""

        class ToolFormat(str, Enum):
            FUNCTION_CALL = "function_call"
            JSON = "json"
            MIXED = "mixed"

        # Simulate MCP build_yaml_pipeline_interactive data
        pipeline = {
            "apiVersion": "hexdag/v1",
            "kind": "Pipeline",
            "metadata": {"name": "test-pipeline"},
            "spec": {
                "nodes": [
                    {
                        "kind": "macro_invocation",
                        "metadata": {"name": "agent"},
                        "spec": {
                            "macro": "core:reasoning_agent",
                            "config": {
                                "main_prompt": "Test prompt",
                                "max_steps": 5,
                                "tool_format": ToolFormat.MIXED,  # ← Enum value
                            },
                        },
                    }
                ]
            },
        }

        # Normalize
        normalized = _normalize_for_yaml(pipeline)

        # Serialize to YAML
        yaml_str = yaml.dump(normalized, sort_keys=False, default_flow_style=False)

        # Deserialize back
        parsed = yaml.safe_load(yaml_str)

        # Verify the enum was serialized as its string value
        tool_format = parsed["spec"]["nodes"][0]["spec"]["config"]["tool_format"]
        assert tool_format == "mixed"
        assert isinstance(tool_format, str)

    def test_non_enum_values_unchanged(self):
        """Test that non-enum values pass through unchanged."""
        input_data = {
            "string": "value",
            "number": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
            "list": [1, 2, 3],
            "dict": {"key": "value"},
        }

        normalized = _normalize_for_yaml(input_data)

        assert normalized == input_data


@pytest.mark.integration
class TestMCPYAMLGeneration:
    """Integration tests for MCP YAML generation with enum handling."""

    def test_build_yaml_pipeline_with_enum_config(self):
        """Test build_yaml_pipeline_interactive with enum values in config."""
        from hexdag.mcp_server import build_yaml_pipeline_interactive

        # Simulate LLM providing enum constant name (the bug case)
        nodes = [
            {
                "kind": "macro_invocation",
                "name": "agent",
                "spec": {
                    "macro": "core:reasoning_agent",
                    "config": {
                        "main_prompt": "Research: {{question}}",
                        "max_steps": 5,
                        "allowed_tools": ["tool1", "tool2"],
                        "tool_format": "MIXED",  # String that looks like enum name
                    },
                },
                "dependencies": [],
            }
        ]

        yaml_output = build_yaml_pipeline_interactive(
            pipeline_name="test-pipeline",
            description="Test pipeline with enums",
            nodes=nodes,
            ports=None,
        )

        # Parse the generated YAML
        parsed = yaml.safe_load(yaml_output)

        # Verify tool_format is preserved as-is (it's already a string)
        tool_format = parsed["spec"]["nodes"][0]["spec"]["config"]["tool_format"]
        assert tool_format == "MIXED"

        # This will fail validation, but that's expected - the MCP server
        # should ideally validate and convert to lowercase "mixed"

    def test_validate_yaml_catches_enum_errors(self):
        """Test that validate_yaml_pipeline catches enum validation errors."""
        from hexdag.mcp_server import validate_yaml_pipeline

        # YAML with incorrect enum value (uppercase instead of lowercase)
        bad_yaml = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-pipeline
spec:
  nodes:
    - kind: macro_invocation
      metadata:
        name: agent
      spec:
        macro: core:reasoning_agent
        config:
          main_prompt: "Test"
          max_steps: 5
          allowed_tools: []
          tool_format: MIXED  # Wrong - should be "mixed"
      dependencies: []
"""

        result = validate_yaml_pipeline(bad_yaml)
        parsed_result = yaml.safe_load(result)

        # Should report validation error
        assert parsed_result["valid"] is False
        assert "tool_format" in parsed_result["error"].lower() or "mixed" in parsed_result[
            "error"
        ].lower()
