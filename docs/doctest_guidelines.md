# Doctest Guidelines for hexDAG

This guide explains how to write doctests that work with our automated testing.

## Configuration

Doctests run with these flags (configured in `pyproject.toml`):
- `NORMALIZE_WHITESPACE`: Ignores differences in whitespace
- `ELLIPSIS`: Use `...` to match any text
- `IGNORE_EXCEPTION_DETAIL`: Ignores exception details in traceback

## Writing Effective Doctests

### 1. Skip Doctests That Can't Run Standalone

Use the `# doctest: +SKIP` directive for examples that:
- Require async context
- Need undefined variables
- Depend on optional dependencies

```python
def my_async_function():
    """
    Examples
    --------
    >>> # doctest: +SKIP
    >>> result = await my_async_function()
    >>> print(result)
    """
```

### 2. Async Code Examples

For async examples, add `# doctest: +SKIP`:

```python
async def fetch_data():
    """
    Examples
    --------
    >>> # doctest: +SKIP
    >>> async with ExecutionContext(...) as ctx:
    ...     data = await fetch_data()
    """
```

### 3. Exception Examples

For examples showing exceptions, use `# doctest: +SKIP`:

```python
class MyError(Exception):
    """
    Examples
    --------
    >>> # doctest: +SKIP
    >>> raise MyError("Something went wrong")
    Traceback (most recent call last):
        ...
    MyError: Something went wrong
    """
```

### 4. Examples with Optional Dependencies

Skip examples requiring optional dependencies:

```python
def configure_openai():
    """
    Examples
    --------
    >>> # doctest: +SKIP
    >>> from hexai.adapters.openai import OpenAIAdapter
    >>> adapter = OpenAIAdapter(api_key="...")
    """
```

### 5. Use Mock Objects for Testable Examples

Prefer using mock adapters for runnable examples:

```python
def create_llm_node():
    """
    Examples
    --------
    >>> from hexai.adapters.mock import MockLLM
    >>> llm = MockLLM()
    >>> # Use llm in your example...
    """
```

### 6. Object Representation

Use `...` for object addresses and dynamic values:

```python
def create_config():
    """
    Examples
    --------
    >>> config = PortConfig(port=MockLLM())
    >>> config.port  # doctest: +ELLIPSIS
    <hexai.adapters.mock.mock_llm.MockLLM object at 0x...>
    """
```

### 7. Setup Code

Use setup sections for complex examples:

```python
def process_data():
    """
    Examples
    --------
    >>> from hexai.adapters.mock import MockLLM
    >>> from hexai.core.domain import NodeSpec
    >>>
    >>> # Setup
    >>> llm = MockLLM()
    >>> spec = NodeSpec(id="test", type="llm", params={})
    >>>
    >>> # Test
    >>> result = process_data(spec, llm)
    >>> result is not None
    True
    """
```

## Running Doctests

### Locally
```bash
# Run all doctests
uv run pytest --doctest-modules hexai/ --ignore=hexai/cli/

# Run with verbose output
uv run pytest --doctest-modules hexai/ --ignore=hexai/cli/ -v

# Continue on failure to see all errors
uv run pytest --doctest-modules hexai/ --ignore=hexai/cli/ --doctest-continue-on-failure
```

### Pre-commit
Doctests run automatically on commit. They use `--doctest-continue-on-failure` to report all issues.

### Azure Pipelines
Doctests run in the "Quality" stage before main tests, generating JUnit XML reports.

## Best Practices

1. **Keep examples simple**: Focus on demonstrating the API, not complex workflows
2. **Use mock adapters**: Prefer `MockLLM`, `MockMemory` over real implementations
3. **Skip when needed**: Don't force async examples to be synchronous
4. **Test what you write**: Run doctests locally before committing
5. **Update failing tests**: If API changes, update the doctest examples

## Common Issues

### `SyntaxError: 'await' outside function`
**Solution**: Add `# doctest: +SKIP` to async examples

### `ModuleNotFoundError: No module named 'hexai.adapters.openai'`
**Solution**: Either skip the example or use mock adapters

### `NameError: name 'foo' is not defined`
**Solution**: Define all variables in the example or skip it

### Object address mismatch
**Solution**: Use `# doctest: +ELLIPSIS` and `0x...` for addresses
