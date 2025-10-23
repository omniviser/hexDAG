"""Unit tests for secure JSON utilities."""

from __future__ import annotations

import json

from pydantic import BaseModel

from hexdag.core.validation.secure_json import SafeJSON


def test_extract_json_from_markdown_fenced_block():
    text = """
    Here is the output:

    ```json
    {"a": 1, "b": 2}
    ```
    """
    extracted = SafeJSON._extract_json(text)
    assert extracted is not None
    assert json.loads(extracted) == {"a": 1, "b": 2}


def test_extract_json_from_any_fenced_block():
    text = """
    Output below:
    ```
    {"x": [1,2,3]}
    ```
    """
    extracted = SafeJSON._extract_json(text)
    assert extracted is not None
    assert json.loads(extracted) == {"x": [1, 2, 3]}


def test_extract_json_by_bracket_matching():
    text = 'Some text before {\n  "k": "v"\n} and after'
    extracted = SafeJSON._extract_json(text)
    assert extracted is not None
    assert json.loads(extracted) == {"k": "v"}


def test_secure_loads_fast_path_valid():
    result = SafeJSON().loads('{"a": 1}')
    assert result.error is None
    assert result.data == {"a": 1}


def test_secure_loads_rejects_large_input():
    # Use a small limit to avoid allocating very large strings in tests
    large_value = "a" * 2000
    large_str = '{"x": "' + large_value + '"}'
    result = SafeJSON(max_size_bytes=1000).loads(large_str)
    assert result.error is not None
    assert result.error == "too_large"


def test_secure_loads_rejects_too_deep_nesting():
    # Construct text with depth DEFAULT_MAX_DEPTH + 5
    sj = SafeJSON()
    depth = sj.max_depth + 5
    nested = "[" * depth + "]" * depth
    result = sj.loads(nested)
    assert result.error is not None
    assert result.error == "too_deep"


def test_secure_loads_cleans_trailing_commas_and_single_quotes():
    dirty = """{ "a": 1, "b": [1,2,], 'c': 'val', }"""  # noqa: E501
    result = SafeJSON().loads(dirty)
    assert result.error is None
    assert result.data == {"a": 1, "b": [1, 2], "c": "val"}


def test_secure_loads_strips_inline_comments():
    dirty = '{"a": 1, // comment\n "b": 2 # another\n }'
    result = SafeJSON().loads(dirty)
    assert result.error is None
    assert result.data == {"a": 1, "b": 2}


def test_secure_loads_returns_none_on_unrecoverable():
    result = SafeJSON().loads('{"a"')
    assert result.error is not None
    assert result.error == "invalid_syntax"


def test_loads_from_llm_output_end_to_end():
    text = """Result
```json
{
  "ok": true, // comment
}
```
"""
    result = SafeJSON().loads_from_text(text)
    assert result.error is None
    assert result.data == {"ok": True}


def test_secure_loader_handles_invalid_json_returns_none():
    class M(BaseModel):
        a: int

    # secure loader should report invalid syntax for unrecoverable JSON
    result = SafeJSON().loads('{"a": not_a_number}')
    assert result.error is not None
    assert result.error == "invalid_syntax"


def test_secure_loader_handles_minor_llm_formatting_and_pydantic_validation():
    class M(BaseModel):
        a: int
        b: list[int]

    dirty = """{ 'a': 1, 'b': [1,2,], }"""
    res = SafeJSON().loads(dirty)
    assert res.error is None
    data = res.data
    assert data == {"a": 1, "b": [1, 2]}

    # Validate via Pydantic directly
    result = M.model_validate(data)
    assert result.a == 1
    assert result.b == [1, 2]
