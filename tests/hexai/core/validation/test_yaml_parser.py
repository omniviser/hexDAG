import pytest
from pydantic import BaseModel

from hexai.core.validation.yaml_parser import SecureYAMLParser


class TestSchema(BaseModel):
    name: str
    age: int


@pytest.fixture
def parser():
    return SecureYAMLParser()


def test_valid_yaml(parser):
    content = "name: Ala\nage: 10"
    result = parser.parse_and_validate(content, TestSchema)
    assert result.ok
    assert result.data.name == "Ala"


def test_yaml_syntax_error(parser):
    bad_yaml = "name: Ala\n  age: 10"
    result = parser.parse_and_validate(bad_yaml, TestSchema)
    assert not result.ok
    assert result.errors and len(result.errors) > 0


def test_extract_from_response(parser):
    text = """```yaml
    name: Olek
    age: 8
    ```"""
    extracted = parser.extract_from_response(text)
    assert extracted.strip().startswith("name:")
