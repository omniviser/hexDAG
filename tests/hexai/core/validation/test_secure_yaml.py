"""Unit tests for secure YAML utilities."""

from __future__ import annotations

import yaml
from pydantic import BaseModel

from hexai.core.validation.secure_yaml import SafeYAML


def test_extract_yaml_from_markdown_fenced_block_yaml():
    text = """
    Here is the output:

    ```yaml
    a: 1
    b:
      - 2
    ```
    """
    extracted = SafeYAML._extract_yaml(text)
    assert extracted is not None
    assert yaml.safe_load(extracted) == {"a": 1, "b": [2]}


def test_extract_yaml_from_markdown_fenced_block_yml():
    text = """
    Output below:

    ```yml
    x:
      - one
      - two
    ```
    """
    extracted = SafeYAML._extract_yaml(text)
    assert extracted is not None
    assert yaml.safe_load(extracted) == {"x": ["one", "two"]}


def test_extract_yaml_by_generic_key_colon_fallback():
    text = """
    Some text before
    order: 42
    items:
      - one
      - two
    and after
    """
    extracted = SafeYAML._extract_yaml(text)
    assert extracted is not None
    assert yaml.safe_load(extracted) == {"order": 42, "items": ["one", "two"]}


def test_secure_loads_fast_path_valid():
    valid = "a: 1\nb:\n  - 1\n  - 2\n"
    result = SafeYAML().loads(valid)
    assert result.error is None
    assert result.data == {"a": 1, "b": [1, 2]}


def test_secure_loads_rejects_large_input():
    # Use a small limit to avoid allocating very large strings in tests
    large_value = "a" * 2000
    large_str = "x: " + large_value
    result = SafeYAML(max_size_bytes=1000).loads(large_str)
    assert result.error is not None
    assert result.error == "too_large"


def test_secure_loads_rejects_too_deep_nesting():
    sj = SafeYAML()
    depth = sj.max_depth + 5
    too_deep_text = (" " * (2 * depth)) + "k: v"
    result = sj.loads(too_deep_text)
    assert result.error is not None
    assert result.error == "too_deep"


def test_secure_loads_returns_invalid_syntax_and_location_preview():
    # Missing closing bracket in flow sequence should raise a marked YAML error
    invalid = "a: 1\nb: [1, 2,\n"
    result = SafeYAML().loads(invalid)
    assert result.error is not None
    assert result.error == "invalid_syntax"
    assert result.line is not None
    assert result.col is not None
    assert result.preview is not None
    assert "^" in result.preview


def test_loads_from_llm_output_end_to_end():
    text = """Result
```yaml
ok: true # comment
list:
  - 1
```
"""
    result = SafeYAML().loads_from_text(text)
    assert result.error is None
    assert result.data == {"ok": True, "list": [1]}


def test_loads_from_text_no_yaml_found():
    text = """
    This text contains no fenced blocks and no YAML-like key lines
    Just plain sentences without any colon patterns
    """
    result = SafeYAML().loads_from_text(text)
    assert result.error is not None
    assert result.error == "no_yaml_found"


def test_secure_loads_accepts_bytes_input():
    data = b"a: 1\nb: 2\n"
    result = SafeYAML().loads(data)
    assert result.error is None
    assert result.data == {"a": 1, "b": 2}


def test_fenced_language_is_case_insensitive():
    text = """
    ```YAML
    x: 1
    ```
    """
    result = SafeYAML().loads_from_text(text)
    assert result.error is None
    assert result.data == {"x": 1}


def test_secure_loader_pydantic_validation():
    class M(BaseModel):
        a: int
        b: list[int]

    yaml_text = "a: 1\nb:\n  - 1\n  - 2\n"
    res = SafeYAML().loads(yaml_text)
    assert res.error is None
    data = res.data
    assert data == {"a": 1, "b": [1, 2]}

    result = M.model_validate(data)
    assert result.a == 1
    assert result.b == [1, 2]
