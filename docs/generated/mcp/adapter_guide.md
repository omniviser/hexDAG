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
from hexdag.kernel.secrets import secret

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
| `MockDatabaseAdapter` | Mock implementation of Database for testing and demos. |
| `MySQLAdapter` | MySQL database adapter with async connection pooling. |
| `PostgreSQLAdapter` | PostgreSQL database adapter with async connection pooling. |
| `SQLAdapter` | Base SQL adapter with SQLAlchemy connection pooling. |
| `SQLiteAdapter` | Async SQLite adapter for database port. |
| `MySQLAdapter` | MySQL document-store adapter for hexDAG. |
| `MySQLAdapter` | MySQL adapter for database port. |

### llm

| Adapter | Description |
|---------|-------------|
| `AnthropicAdapter` | Anthropic implementation of the LLM port. |
| `AzureOpenAIAdapter` | Azure OpenAI adapter for LLM port with embedding support. |
| `LLM` | Port interface for Large Language Models (LLMs). |
| `MockLLM` | Mock implementation of the LLM interface for testing. |
| `OpenAIAdapter` | Unified OpenAI implementation of the LLM port. |

### memory

| Adapter | Description |
|---------|-------------|
| `AzureCosmosAdapter` | Azure Cosmos DB adapter for agent memory and pipeline state. |
| `InMemoryMemory` | In-memory implementation of Memory for testing. |
| `Memory` | Protocol for long-term memory storage and retrieval. |
| `FileMemoryAdapter` | Memory adapter backed by file system. |
| `SQLiteMemoryAdapter` | Memory adapter backed by SQLite database. |

### secret

| Adapter | Description |
|---------|-------------|
| `AzureKeyVaultAdapter` | Azure Key Vault adapter for secret resolution. |
| `LocalSecretAdapter` | Local secret adapter that reads from environment variables. |

### storage

| Adapter | Description |
|---------|-------------|
| `AzureBlobAdapter` | Azure Blob Storage adapter for file operations. |

### unknown

| Adapter | Description |
|---------|-------------|
| `PgVectorAdapter` | PostgreSQL adapter with pgvector extension support. |
| `ChromaDBAdapter` | ChromaDB vector store adapter. |
| `HexDAGAdapter` | Mixin that auto-registers adapters when ``yaml_alias`` is pr... |
| `PgVectorAdapter` | PostgreSQL pgvector adapter using SQLAlchemy. |

## Using Adapters in YAML

```yaml
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: my-pipeline
spec:
  ports:
    llm:
      adapter: hexdag.stdlib.adapters.openai.OpenAIAdapter
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
