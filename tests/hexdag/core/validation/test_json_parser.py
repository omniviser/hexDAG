import pytest
from pydantic import BaseModel, Field

from hexdag.core.validation.json_parser import SecureJSONParser


class TestSchema(BaseModel):
    name: str
    age: int = Field(ge=0)


@pytest.fixture
def parser():
    return SecureJSONParser()


def test_valid_json(parser):
    content = '{"name": "Ala", "age": 10}'
    result = parser.parse_and_validate(content, TestSchema)
    assert result.ok
    assert result.data.name == "Ala"
    assert result.data.age == 10


def test_invalid_json_syntax(parser):
    content = '{"name": "Ala", "age": }'
    result = parser.parse_and_validate(content, TestSchema)
    assert not result.ok
    assert any("invalid" in e.lower() or "expect" in e.lower() for e in result.errors)


def test_validation_error(parser):
    content = '{"name": "Ala", "age": -3}'
    result = parser.parse_and_validate(content, TestSchema)
    assert not result.ok
    assert "ge" in result.errors[0] or "negative" in result.errors[0].lower()


def test_extract_from_response(parser):
    response = """Here you go:
    ```json
    {"name": "Olek", "age": 12}
    ```"""
    extracted = parser.extract_from_response(response)
    assert extracted.startswith("{")
    result = parser.parse_and_validate(extracted, TestSchema)
    assert result.ok
