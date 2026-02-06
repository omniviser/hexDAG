# Creating Custom Adapters in hexDAG

## Overview

hexDAG uses adapters to connect pipelines to external services like LLMs,
databases, and APIs. Adapters implement ports (interfaces) with async methods.

## Quick Start

### Simple Adapter (No Secrets)

```python
class MemoryCacheAdapter:
    """Simple in-memory cache adapter."""

    def __init__(self, max_size: int = 100, ttl: int = 3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl

    async def aget(self, key: str):
        return self.cache.get(key)

    async def aset(self, key: str, value: any):
        self.cache[key] = value
```

### Adapter with Secrets

Use `secret()` in defaults to declare secrets that auto-resolve from environment:

```python
from hexdag.core.secrets import secret

class OpenAIAdapter:
    """OpenAI LLM adapter with automatic secret resolution."""

    def __init__(
        self,
        api_key: str = secret(env="OPENAI_API_KEY"),  # Auto-resolved
        model: str = "gpt-4",
        temperature: float = 0.7
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

    async def aresponse(self, messages: list) -> str:
        # Your implementation using self.api_key
        ...
```

## Secret Resolution

Secrets declared with `secret()` are resolved in this order:
1. **Explicit kwargs** - Values passed directly to `__init__`
2. **Environment variables** - From the `env` parameter
3. **Memory port** - From orchestrator memory (with `secret:` prefix)
4. **Error** - If required and no default

## Available Adapters

### database

| Adapter | Description |
|---------|-------------|
| `SQLiteAdapter` | Async SQLite adapter for database port. |
| `CsvAdapter` | Adapter class for reading CSV files from a specified directo... |
| `MockDatabaseAdapter` | Mock implementation of DatabasePort for testing and demos. |

### llm

| Adapter | Description |
|---------|-------------|
| `OpenAIAdapter` | Unified OpenAI implementation of the LLM port. |
| `AnthropicAdapter` | Anthropic implementation of the LLM port. |

### memory

| Adapter | Description |
|---------|-------------|
| `InMemoryMemory` | In-memory implementation of Memory for testing. |
| `Memory` | Protocol for long-term memory storage and retrieval. |
| `SQLiteMemoryAdapter` | Memory adapter backed by SQLite database. |
| `FileMemoryAdapter` | Memory adapter backed by file system. |
| `Memory` | Protocol for long-term memory storage and retrieval. |
| `Memory` | Protocol for long-term memory storage and retrieval. |

### observer_manager

| Adapter | Description |
|---------|-------------|
| `LocalObserverManager` | Local standalone implementation of observer manager. |

### secret

| Adapter | Description |
|---------|-------------|
| `LocalSecretAdapter` | Local secret adapter that reads from environment variables. |

### tool_router

| Adapter | Description |
|---------|-------------|
| `MockToolRouter` | Mock implementation of ToolRouter for testing. |
| `ToolRouter` | Protocol for routing tool calls. |
| `ToolRouter` | Protocol for routing tool calls. |
| `UnifiedToolRouter` | ToolRouter adapter that supports multiple tool sources with ... |

## Using Adapters in YAML

```yaml
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: my-pipeline
spec:
  ports:
    llm:
      adapter: hexdag.builtin.adapters.openai.OpenAIAdapter
      config:
        api_key: ${OPENAI_API_KEY}
        model: gpt-4

  nodes:
    - kind: llm_node
      metadata:
        name: analyzer
      spec:
        prompt_template: "Analyze: {{input}}"
      dependencies: []
```

## Best Practices

1. **Async First**: Use `async def` for I/O operations
2. **Type Hints**: Add type annotations for better tooling
3. **Docstrings**: Document your adapter's purpose and config
4. **Error Handling**: Wrap external calls in try/except
5. **Secrets**: Use `secret()` - never hardcode secrets
