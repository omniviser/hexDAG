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


# ── YAML parsing tests ──────────────────────────────────────────────


def test_loads_yaml_simple():
    text = "key: value\ncount: 42\n"
    result = SafeJSON().loads_yaml(text)
    assert result.ok
    assert result.data == {"key": "value", "count": 42}


def test_loads_yaml_rejects_large_input():
    large = "x: " + "a" * 2000
    result = SafeJSON(max_size_bytes=1000).loads_yaml(large)
    assert result.error == "too_large"


def test_loads_yaml_invalid_yaml():
    result = SafeJSON().loads_yaml(":::\n  - ][bad")
    assert result.error == "yaml_error"


def test_loads_yaml_nested():
    text = """
outer:
  inner:
    value: 1
  list:
    - a
    - b
"""
    result = SafeJSON().loads_yaml(text)
    assert result.ok
    assert result.data["outer"]["inner"]["value"] == 1
    assert result.data["outer"]["list"] == ["a", "b"]


def test_loads_yaml_from_text_fenced_block():
    text = """Here is the config:
```yaml
name: test
count: 5
```
"""
    result = SafeJSON().loads_yaml_from_text(text)
    assert result.ok
    assert result.data == {"name": "test", "count": 5}


def test_loads_yaml_from_text_yml_block():
    text = """Output:
```yml
items:
  - one
  - two
```
"""
    result = SafeJSON().loads_yaml_from_text(text)
    assert result.ok
    assert result.data["items"] == ["one", "two"]


def test_loads_yaml_from_text_generic_block():
    text = """Result:
```
key: value
```
"""
    result = SafeJSON().loads_yaml_from_text(text)
    assert result.ok
    assert result.data == {"key": "value"}


def test_loads_yaml_from_text_raw():
    text = "name: test\ncount: 42"
    result = SafeJSON().loads_yaml_from_text(text)
    assert result.ok
    assert result.data == {"name": "test", "count": 42}


# ── _extract_yaml tests ─────────────────────────────────────────────


def test_extract_yaml_from_yaml_block():
    text = "```yaml\nfoo: bar\n```"
    assert SafeJSON._extract_yaml(text) == "foo: bar"


def test_extract_yaml_from_yml_block():
    text = "```yml\nfoo: bar\n```"
    assert SafeJSON._extract_yaml(text) == "foo: bar"


def test_extract_yaml_from_generic_block():
    text = "```\nfoo: bar\n```"
    assert SafeJSON._extract_yaml(text) == "foo: bar"


def test_extract_yaml_raw_text():
    assert SafeJSON._extract_yaml("foo: bar") == "foo: bar"


def test_extract_yaml_empty():
    assert SafeJSON._extract_yaml("") == ""


# ── loads_from_text integration (JSON strategies) ────────────────────


def test_loads_from_text_plain_json():
    result = SafeJSON().loads_from_text('{"key": "value"}')
    assert result.ok
    assert result.data == {"key": "value"}


def test_loads_from_text_json_in_markdown():
    text = 'Here is JSON:\n```json\n{"a": 1}\n```\n'
    result = SafeJSON().loads_from_text(text)
    assert result.ok
    assert result.data == {"a": 1}


def test_loads_from_text_json_with_comments_in_markdown():
    text = '```json\n{"a": 1, // comment\n"b": 2}\n```'
    result = SafeJSON().loads_from_text(text)
    assert result.ok
    assert result.data == {"a": 1, "b": 2}


def test_loads_from_text_no_json():
    result = SafeJSON().loads_from_text("just plain text with no JSON")
    assert result.error == "no_json_found"


def test_loads_from_text_array():
    result = SafeJSON().loads_from_text("Result: [1, 2, 3]")
    assert result.ok
    assert result.data == [1, 2, 3]
