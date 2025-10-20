import pytest
from pydantic import BaseModel, Field

from hexai.core.validation.unified_engine import UnifiedParsingEngine


class ExampleSchema(BaseModel):
    name: str
    age: int = Field(ge=0)


@pytest.fixture
def engine():
    return UnifiedParsingEngine()


def test_auto_detect_json(engine):
    content = """```json
    {"name": "Ola", "age": 5}
    ```"""
    result = engine.auto_detect_and_parse(content, ExampleSchema)
    assert result.ok
    assert result.data.name == "Ola"


def test_auto_detect_yaml(engine):
    content = """```yaml
    name: Tomek
    age: 7
    ```"""
    result = engine.auto_detect_and_parse(content, ExampleSchema)
    assert result.ok
    assert result.data.age == 7


def test_invalid_input(engine):
    bad = "not a valid structured data"
    result = engine.auto_detect_and_parse(bad, ExampleSchema)
    assert not result.ok
