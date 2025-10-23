"""Tests for ParserNode - structured output parsing."""

import pytest
from pydantic import BaseModel

from hexdag.builtin.nodes.parser_node import ParserNode
from hexdag.core.exceptions import ParseError


class Person(BaseModel):
    """Test model for parsing."""

    name: str
    age: int


class TestParserNode:
    """Test ParserNode functionality."""

    def test_create_parser_node(self):
        """Test creating a parser node."""
        parser = ParserNode()
        spec = parser(name="person_parser", output_schema=Person)

        assert spec.name == "person_parser"
        assert spec.fn is not None

    @pytest.mark.asyncio
    async def test_parse_valid_json(self):
        """Test parsing valid JSON."""
        parser = ParserNode()
        spec = parser(name="parser", output_schema=Person)

        json_text = '{"name": "Alice", "age": 30}'
        result = await spec.fn({"text": json_text})

        assert isinstance(result, Person)
        assert result.name == "Alice"
        assert result.age == 30

    @pytest.mark.asyncio
    async def test_parse_json_with_whitespace(self):
        """Test parsing JSON with extra whitespace."""
        parser = ParserNode()
        spec = parser(name="parser", output_schema=Person)

        json_text = '  \n  {"name": "Bob", "age": 25}  \n  '
        result = await spec.fn({"text": json_text})

        assert result.name == "Bob"
        assert result.age == 25

    @pytest.mark.asyncio
    async def test_parse_json_in_text(self):
        """Test extracting JSON from surrounding text."""
        parser = ParserNode()
        spec = parser(name="parser", output_schema=Person)

        text = 'Here is the data: {"name": "Charlie", "age": 35} - end'
        result = await spec.fn({"text": text})

        assert result.name == "Charlie"
        assert result.age == 35

    @pytest.mark.asyncio
    async def test_parse_json_in_markdown(self):
        """Test extracting JSON from markdown code blocks."""
        parser = ParserNode()
        spec = parser(name="parser", output_schema=Person, strategy="json_in_markdown")

        text = """
Here's the result:

```json
{
    "name": "David",
    "age": 40
}
```

That's the data.
"""
        result = await spec.fn({"text": text})

        assert result.name == "David"
        assert result.age == 40

    @pytest.mark.asyncio
    async def test_parse_failure_strict_mode(self):
        """Test parse failure in strict mode raises error."""
        parser = ParserNode()
        spec = parser(name="parser", output_schema=Person, strict=True)

        invalid_json = "not json at all"

        with pytest.raises(ParseError) as exc_info:
            await spec.fn({"text": invalid_json})

        assert "Failed to parse" in str(exc_info.value)
        assert "Retry hints" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_parse_failure_non_strict_mode(self):
        """Test parse failure in non-strict mode returns empty model."""
        parser = ParserNode()
        spec = parser(name="parser", output_schema=Person, strict=False)

        invalid_json = "not json at all"
        result = await spec.fn({"text": invalid_json})

        # Should return empty/default Person model
        assert isinstance(result, Person)

    @pytest.mark.asyncio
    async def test_validation_failure_strict_mode(self):
        """Test validation failure in strict mode."""
        parser = ParserNode()
        spec = parser(name="parser", output_schema=Person, strict=True)

        # Valid JSON but wrong schema (age is string instead of int)
        json_text = '{"name": "Eve", "age": "thirty"}'

        with pytest.raises(ParseError) as exc_info:
            await spec.fn({"text": json_text})

        error_msg = str(exc_info.value)
        assert "does not match expected schema" in error_msg
        assert "Person" in error_msg

    @pytest.mark.asyncio
    async def test_parse_dict_schema(self):
        """Test parsing with dict schema instead of Pydantic model."""
        parser = ParserNode()
        spec = parser(name="parser", output_schema={"name": str, "age": int})

        json_text = '{"name": "Frank", "age": 45}'
        result = await spec.fn({"text": json_text})

        assert isinstance(result, BaseModel)
        assert result.name == "Frank"  # type: ignore[attr-defined]
        assert result.age == 45  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_parse_with_string_input(self):
        """Test parsing when input is a string instead of dict."""
        parser = ParserNode()
        spec = parser(name="parser", output_schema=Person)

        json_text = '{"name": "Grace", "age": 28}'
        # Pass string directly instead of dict
        result = await spec.fn(json_text)

        assert result.name == "Grace"
        assert result.age == 28

    @pytest.mark.asyncio
    async def test_error_message_quality(self):
        """Test that error messages are helpful for retry logic."""
        parser = ParserNode()
        spec = parser(name="parser", output_schema=Person, strict=True)

        invalid_json = "This is not JSON: {name: Alice, age: 30}"

        try:
            await spec.fn({"text": invalid_json})
            pytest.fail("Should have raised ParseError")
        except ParseError as e:
            error_msg = str(e)

            # Check for helpful retry hints
            assert "Retry hints" in error_msg
            assert "Output preview" in error_msg
            assert "properly formatted" in error_msg

    @pytest.mark.asyncio
    async def test_parse_markdown_without_language_tag(self):
        """Test parsing JSON from markdown without language tag."""
        parser = ParserNode()
        spec = parser(name="parser", output_schema=Person, strategy="json_in_markdown")

        text = """
```
{"name": "Henry", "age": 50}
```
"""
        result = await spec.fn({"text": text})

        assert result.name == "Henry"
        assert result.age == 50
