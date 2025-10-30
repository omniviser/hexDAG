"""Tests for SchemaGenerator."""

import json
from typing import Annotated, Literal

import pytest
import yaml
from pydantic import Field

from hexdag.core.schema.generator import SchemaGenerator


class TestBasicTypes:
    """Test basic type conversion."""

    def test_string_type(self):
        """Test string type conversion."""

        def factory(value: str):
            pass

        schema = SchemaGenerator.from_callable(factory)
        assert schema["properties"]["value"]["type"] == "string"

    def test_integer_type(self):
        """Test integer type conversion."""

        def factory(count: int):
            pass

        schema = SchemaGenerator.from_callable(factory)
        assert schema["properties"]["count"]["type"] == "integer"

    def test_float_type(self):
        """Test float type conversion."""

        def factory(value: float):
            pass

        schema = SchemaGenerator.from_callable(factory)
        assert schema["properties"]["value"]["type"] == "number"

    def test_boolean_type(self):
        """Test boolean type conversion."""

        def factory(enabled: bool):
            pass

        schema = SchemaGenerator.from_callable(factory)
        assert schema["properties"]["enabled"]["type"] == "boolean"

    def test_list_type(self):
        """Test list type conversion."""

        def factory(items: list[str]):
            pass

        schema = SchemaGenerator.from_callable(factory)
        assert schema["properties"]["items"]["type"] == "array"
        assert schema["properties"]["items"]["items"]["type"] == "string"

    def test_dict_type(self):
        """Test dict type conversion."""

        def factory(config: dict):
            pass

        schema = SchemaGenerator.from_callable(factory)
        assert schema["properties"]["config"]["type"] == "object"


class TestLiteralTypes:
    """Test Literal type conversion to enum."""

    def test_literal_string_enum(self):
        """Test Literal strings become enum."""

        def factory(size: Literal["small", "medium", "large"]):
            pass

        schema = SchemaGenerator.from_callable(factory)
        assert schema["properties"]["size"]["enum"] == ["small", "medium", "large"]
        assert schema["properties"]["size"]["type"] == "string"

    def test_literal_integer_enum(self):
        """Test Literal integers become enum."""

        def factory(level: Literal[1, 2, 3]):
            pass

        schema = SchemaGenerator.from_callable(factory)
        assert schema["properties"]["level"]["enum"] == [1, 2, 3]
        assert schema["properties"]["level"]["type"] == "integer"


class TestUnionTypes:
    """Test Union type conversion."""

    def test_union_types(self):
        """Test Union types become anyOf."""

        def factory(value: str | int):
            pass

        schema = SchemaGenerator.from_callable(factory)
        assert "anyOf" in schema["properties"]["value"]
        types = [s["type"] for s in schema["properties"]["value"]["anyOf"]]
        assert "string" in types
        assert "integer" in types

    def test_optional_type(self):
        """Test Optional[T] becomes nullable."""

        def factory(value: str | None = None):
            pass

        schema = SchemaGenerator.from_callable(factory)
        # Should be string with null allowed
        prop = schema["properties"]["value"]
        assert "type" in prop
        # Type should include null
        if isinstance(prop["type"], list):
            assert "null" in prop["type"]


class TestAnnotatedTypes:
    """Test Annotated types with Field constraints."""

    def test_field_ge_constraint(self):
        """Test Field with ge (greater than or equal) constraint."""

        def factory(value: Annotated[int, Field(ge=0)]):
            pass

        schema = SchemaGenerator.from_callable(factory)
        assert schema["properties"]["value"]["minimum"] == 0

    def test_field_le_constraint(self):
        """Test Field with le (less than or equal) constraint."""

        def factory(value: Annotated[int, Field(le=100)]):
            pass

        schema = SchemaGenerator.from_callable(factory)
        assert schema["properties"]["value"]["maximum"] == 100

    def test_field_ge_le_constraints(self):
        """Test Field with both ge and le constraints."""

        def factory(temperature: Annotated[float, Field(ge=0.0, le=2.0)]):
            pass

        schema = SchemaGenerator.from_callable(factory)
        assert schema["properties"]["temperature"]["minimum"] == 0.0
        assert schema["properties"]["temperature"]["maximum"] == 2.0
        assert schema["properties"]["temperature"]["type"] == "number"

    def test_field_description(self):
        """Test Field with description."""

        def factory(value: Annotated[str, Field(description="A test value")]):
            pass

        schema = SchemaGenerator.from_callable(factory)
        assert schema["properties"]["value"]["description"] == "A test value"


class TestDefaults:
    """Test default value handling."""

    def test_default_values(self):
        """Test parameters with defaults."""

        def factory(value: str, count: int = 10, enabled: bool = True):
            pass

        schema = SchemaGenerator.from_callable(factory)

        # Required field (no default)
        assert "value" in schema["required"]

        # Optional fields (have defaults)
        assert "count" not in schema.get("required", [])
        assert "enabled" not in schema.get("required", [])

        # Defaults are stored
        assert schema["properties"]["count"]["default"] == 10
        assert schema["properties"]["enabled"]["default"] is True

    def test_no_required_when_all_defaults(self):
        """Test no required field when all params have defaults."""

        def factory(name: str = "default", count: int = 0):
            pass

        schema = SchemaGenerator.from_callable(factory)
        assert schema.get("required", []) == []


class TestDocstringExtraction:
    """Test docstring extraction for descriptions."""

    def test_google_style_docstring(self):
        """Test Google-style docstring extraction."""

        def factory(value: str, count: int):
            """Example function.

            Args:
                value: The value parameter
                count: The count parameter
            """
            pass

        schema = SchemaGenerator.from_callable(factory)
        assert schema["properties"]["value"]["description"] == "The value parameter"
        assert schema["properties"]["count"]["description"] == "The count parameter"

    def test_numpy_style_docstring(self):
        """Test NumPy-style docstring extraction."""

        def factory(value: float):
            """Example function.

            Parameters
            ----------
            value: The value parameter
            """
            pass

        schema = SchemaGenerator.from_callable(factory)
        assert schema["properties"]["value"]["description"] == "The value parameter"

    def test_no_docstring(self):
        """Test function without docstring."""

        def factory(value: str):
            pass

        schema = SchemaGenerator.from_callable(factory)
        # Should not have description
        assert "description" not in schema["properties"]["value"]


class TestOutputFormats:
    """Test different output formats."""

    def test_dict_format(self):
        """Test dict format (default)."""

        def factory(name: str):
            pass

        schema = SchemaGenerator.from_callable(factory, format="dict")
        assert isinstance(schema, dict)
        assert schema["type"] == "object"

    def test_yaml_format(self):
        """Test YAML format output."""

        def factory(value: str):
            pass

        schema_yaml = SchemaGenerator.from_callable(factory, format="yaml")
        assert isinstance(schema_yaml, str)

        # Should be valid YAML
        parsed = yaml.safe_load(schema_yaml)
        assert parsed["type"] == "object"
        assert "value" in parsed["properties"]

    def test_json_format(self):
        """Test JSON format output."""

        def factory(value: str):
            pass

        schema_json = SchemaGenerator.from_callable(factory, format="json")
        assert isinstance(schema_json, str)

        # Should be valid JSON
        parsed = json.loads(schema_json)
        assert parsed["type"] == "object"
        assert "value" in parsed["properties"]

    def test_invalid_format(self):
        """Test invalid format raises error."""

        def factory(name: str):
            pass

        with pytest.raises(ValueError, match="Invalid format"):
            SchemaGenerator.from_callable(factory, format="xml")


class TestExampleGeneration:
    """Test YAML example generation."""

    def test_basic_example(self):
        """Test basic example generation."""
        schema = {
            "properties": {
                "template": {"type": "string"},
                "model": {"type": "string", "default": "gpt-4"},
            },
            "required": ["template"],
        }

        example = SchemaGenerator.generate_example_yaml("llm_node", schema)
        parsed = yaml.safe_load(example)

        assert parsed["kind"] == "llm_node"
        assert parsed["metadata"]["name"] == "my_llm_node"
        assert "template" in parsed["spec"]
        assert parsed["spec"]["model"] == "gpt-4"

    def test_example_with_placeholders(self):
        """Test example uses placeholders for required fields."""
        schema = {
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"},
            },
            "required": ["name", "count"],
        }

        example = SchemaGenerator.generate_example_yaml("test_node", schema)
        parsed = yaml.safe_load(example)

        # Required fields should have placeholders
        assert parsed["spec"]["name"] == "value"
        assert parsed["spec"]["count"] == 0


class TestComplexScenarios:
    """Test complex real-world scenarios."""

    def test_llm_node_like_signature(self):
        """Test signature similar to LLMNode."""

        def llm_node(
            name: str,
            template: str,
            model: str = "gpt-4",
            temperature: Annotated[float, Field(ge=0.0, le=2.0)] = 0.7,
            max_tokens: int | None = None,
        ):
            """Create an LLM node.

            Args:
                name: Node name
                template: Prompt template
                model: Model name
                temperature: Sampling temperature
                max_tokens: Maximum tokens to generate
            """
            pass

        schema = SchemaGenerator.from_callable(llm_node)

        # Check required fields (only template, since name is skipped)
        assert "template" in schema["required"]
        assert "model" not in schema["required"]
        assert "temperature" not in schema["required"]

        # Check types
        assert schema["properties"]["template"]["type"] == "string"
        assert schema["properties"]["model"]["type"] == "string"
        assert schema["properties"]["temperature"]["type"] == "number"

        # Check constraints
        assert schema["properties"]["temperature"]["minimum"] == 0.0
        assert schema["properties"]["temperature"]["maximum"] == 2.0

        # Check defaults
        assert schema["properties"]["model"]["default"] == "gpt-4"
        assert schema["properties"]["temperature"]["default"] == 0.7

        # Check descriptions
        assert schema["properties"]["template"]["description"] == "Prompt template"
        assert schema["properties"]["temperature"]["description"] == "Sampling temperature"

    def test_agent_node_like_signature(self):
        """Test signature similar to AgentNode."""

        def agent_node(
            name: str,
            main_prompt: str,
            max_steps: Annotated[int, Field(ge=1, le=100)] = 20,
            tools: list[str] | None = None,
        ):
            """Create an agent node.

            Args:
                name: Node name
                main_prompt: Initial reasoning prompt
                max_steps: Maximum reasoning steps
                tools: List of tool names
            """
            pass

        schema = SchemaGenerator.from_callable(agent_node)

        # Check required field
        assert "main_prompt" in schema["required"]

        # Check array type (with null since it's optional)
        assert schema["properties"]["tools"]["type"] in (["array", "null"], "array")

        # Check constraints
        assert schema["properties"]["max_steps"]["minimum"] == 1
        assert schema["properties"]["max_steps"]["maximum"] == 100

    def test_skips_special_parameters(self):
        """Test that special parameters are skipped."""

        class MyNode:
            def __call__(
                self,
                name: str,
                template: str,
                model: str = "gpt-4",
            ):
                pass

        schema = SchemaGenerator.from_callable(MyNode())

        # 'self' and 'name' should be skipped
        assert "self" not in schema["properties"]
        assert "name" not in schema["properties"]

        # Other params should be present
        assert "template" in schema["properties"]
        assert "model" in schema["properties"]
