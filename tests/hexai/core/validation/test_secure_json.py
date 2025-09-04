"""Unit tests for secure JSON utilities."""

from __future__ import annotations

import json

import pytest

from hexai.core.validation import JsonStringToPydanticConverter
from hexai.core.validation.secure_json import (
    DEFAULT_MAX_DEPTH,
    extract_json_from_text,
    loads,
    loads_from_llm_output,
)


def test_extract_json_from_markdown_fenced_block():
    text = """
    Here is the output:

    ```json
    {"a": 1, "b": 2}
    ```
    """
    extracted = extract_json_from_text(text)
    assert extracted is not None
    assert json.loads(extracted) == {"a": 1, "b": 2}


def test_extract_json_from_any_fenced_block():
    text = """
    Output below:
    ```
    {"x": [1,2,3]}
    ```
    """
    extracted = extract_json_from_text(text)
    assert extracted is not None
    assert json.loads(extracted) == {"x": [1, 2, 3]}


def test_extract_json_by_bracket_matching():
    text = 'Some text before {\n  "k": "v"\n} and after'
    extracted = extract_json_from_text(text)
    assert extracted is not None
    assert json.loads(extracted) == {"k": "v"}


def test_secure_loads_fast_path_valid():
    data = loads('{"a": 1}')
    assert data == {"a": 1}


def test_secure_loads_rejects_large_input():
    # Use a small limit to avoid allocating very large strings in tests
    large_value = "a" * 2000
    large_str = '{"x": "' + large_value + '"}'
    assert loads(large_str, max_size_bytes=1000) is None


def test_secure_loads_rejects_too_deep_nesting():
    # Construct text with depth DEFAULT_MAX_DEPTH + 5
    depth = DEFAULT_MAX_DEPTH + 5
    nested = "[" * depth + "]" * depth
    assert loads(nested) is None


def test_secure_loads_cleans_trailing_commas_and_single_quotes():
    dirty = """{ "a": 1, "b": [1,2,], 'c': 'val', }"""  # noqa: E501
    data = loads(dirty)
    assert data == {"a": 1, "b": [1, 2], "c": "val"}


def test_secure_loads_strips_inline_comments():
    dirty = '{"a": 1, // comment\n "b": 2 # another\n }'
    data = loads(dirty)
    assert data == {"a": 1, "b": 2}


def test_secure_loads_returns_none_on_unrecoverable():
    assert loads('{"a"') is None


def test_loads_from_llm_output_end_to_end():
    text = """Result
```json
{
  "ok": true, // comment
}
```
"""
    parsed = loads_from_llm_output(text)
    assert parsed == {"ok": True}


def test_converter_uses_secure_loader_and_handles_invalid_json():
    from pydantic import BaseModel

    class M(BaseModel):
        a: int

    conv = JsonStringToPydanticConverter()
    with pytest.raises(Exception):
        conv.convert('{"a": not_a_number}', M)  # invalid JSON


def test_converter_handles_minor_llm_formatting():
    from pydantic import BaseModel

    class M(BaseModel):
        a: int
        b: list[int]

    conv = JsonStringToPydanticConverter()
    dirty = """{ 'a': 1, 'b': [1,2,], }"""
    result = conv.convert(dirty, M)
    assert result.a == 1
    assert result.b == [1, 2]
