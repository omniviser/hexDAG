# Adapters Reference

This page documents all registered adapters in HexDAG.

## Overview

Total adapters: **39**

| Alias | Module Path |
|-------|-------------|
| `AnthropicAdapter` | `hexdag.stdlib.adapters.anthropic.AnthropicAdapter` |
| `CsvAdapter` | `hexdag.stdlib.adapters.database.csv.CsvAdapter` |
| `FileMemoryAdapter` | `hexdag.stdlib.adapters.memory.FileMemoryAdapter` |
| `InMemoryMemory` | `hexdag.stdlib.adapters.memory.InMemoryMemory` |
| `LocalSecretAdapter` | `hexdag.stdlib.adapters.secret.LocalSecretAdapter` |
| `MockDatabaseAdapter` | `hexdag.stdlib.adapters.mock.MockDatabaseAdapter` |
| `MockEmbedding` | `hexdag.stdlib.adapters.mock.MockEmbedding` |
| `MockLLM` | `hexdag.stdlib.adapters.mock.MockLLM` |
| `OpenAIAdapter` | `hexdag.stdlib.adapters.openai.OpenAIAdapter` |
| `PgVectorAdapter` | `hexdag.stdlib.adapters.database.pgvector.PgVectorAdapter` |
| `SQLAlchemyAdapter` | `hexdag.stdlib.adapters.database.sqlalchemy.SQLAlchemyAdapter` |
| `SQLiteAdapter` | `hexdag.stdlib.adapters.database.sqlite.SQLiteAdapter` |
| `SQLiteMemoryAdapter` | `hexdag.stdlib.adapters.memory.SQLiteMemoryAdapter` |
| `anthropic_adapter` | `hexdag.stdlib.adapters.anthropic.AnthropicAdapter` |
| `csv_adapter` | `hexdag.stdlib.adapters.database.csv.CsvAdapter` |
| `database:csv` | `hexdag.stdlib.adapters.database.csv.CsvAdapter` |
| `database:mock` | `hexdag.stdlib.adapters.mock.MockDatabaseAdapter` |
| `database:pgvector` | `hexdag.stdlib.adapters.database.pgvector.PgVectorAdapter` |
| `database:sqlalchemy` | `hexdag.stdlib.adapters.database.sqlalchemy.SQLAlchemyAdapter` |
| `database:sqlite` | `hexdag.stdlib.adapters.database.sqlite.SQLiteAdapter` |
| `embedding:mock` | `hexdag.stdlib.adapters.mock.MockEmbedding` |
| `file_memory_adapter` | `hexdag.stdlib.adapters.memory.FileMemoryAdapter` |
| `in_memory_memory` | `hexdag.stdlib.adapters.memory.InMemoryMemory` |
| `llm:anthropic` | `hexdag.stdlib.adapters.anthropic.AnthropicAdapter` |
| `llm:mock` | `hexdag.stdlib.adapters.mock.MockLLM` |
| `llm:openai` | `hexdag.stdlib.adapters.openai.OpenAIAdapter` |
| `local_secret_adapter` | `hexdag.stdlib.adapters.secret.LocalSecretAdapter` |
| `memory:file` | `hexdag.stdlib.adapters.memory.FileMemoryAdapter` |
| `memory:in_memory` | `hexdag.stdlib.adapters.memory.InMemoryMemory` |
| `memory:sqlite` | `hexdag.stdlib.adapters.memory.SQLiteMemoryAdapter` |
| `mock_database_adapter` | `hexdag.stdlib.adapters.mock.MockDatabaseAdapter` |
| `mock_embedding` | `hexdag.stdlib.adapters.mock.MockEmbedding` |
| `mock_llm` | `hexdag.stdlib.adapters.mock.MockLLM` |
| `open_ai_adapter` | `hexdag.stdlib.adapters.openai.OpenAIAdapter` |
| `pg_vector_adapter` | `hexdag.stdlib.adapters.database.pgvector.PgVectorAdapter` |
| `secret:local` | `hexdag.stdlib.adapters.secret.LocalSecretAdapter` |
| `sq_lite_adapter` | `hexdag.stdlib.adapters.database.sqlite.SQLiteAdapter` |
| `sq_lite_memory_adapter` | `hexdag.stdlib.adapters.memory.SQLiteMemoryAdapter` |
| `sql_alchemy_adapter` | `hexdag.stdlib.adapters.database.sqlalchemy.SQLAlchemyAdapter` |

---

### `AnthropicAdapter`

**Module:** `hexdag.stdlib.adapters.anthropic.AnthropicAdapter`

Anthropic implementation of the LLM port.

**Parameters:**

- **`args`**: `Any`
- **`kwargs`**: `Any`

---

### `CsvAdapter`

**Module:** `hexdag.stdlib.adapters.database.csv.CsvAdapter`

*Could not load component: Cannot resolve 'hexdag.stdlib.adapters.database.csv.CsvAdapter': Class 'CsvAdapter' not found in 'hexdag.stdlib.adapters.database.csv'. Available: *

---

### `FileMemoryAdapter`

**Module:** `hexdag.stdlib.adapters.memory.FileMemoryAdapter`

Memory adapter backed by file system.

**Parameters:**

- **`args`**: `Any`
- **`kwargs`**: `Any`

---

### `InMemoryMemory`

**Module:** `hexdag.stdlib.adapters.memory.InMemoryMemory`

In-memory implementation of Memory for testing.

**Parameters:**

- **`args`**: `Any`
- **`kwargs`**: `Any`

---

### `LocalSecretAdapter`

**Module:** `hexdag.stdlib.adapters.secret.LocalSecretAdapter`

Local secret adapter that reads from environment variables.

**Parameters:**

- **`args`**: `Any`
- **`kwargs`**: `Any`

---

### `MockDatabaseAdapter`

**Module:** `hexdag.stdlib.adapters.mock.MockDatabaseAdapter`

Mock implementation of Database for testing and demos.

**Parameters:**

- **`args`**: `Any`
- **`kwargs`**: `Any`

---

### `MockEmbedding`

**Module:** `hexdag.stdlib.adapters.mock.MockEmbedding`

Mock implementation of the Embedding interface for testing.

**Parameters:**

- **`args`**: `Any`
- **`kwargs`**: `Any`

---

### `MockLLM`

**Module:** `hexdag.stdlib.adapters.mock.MockLLM`

Mock implementation of the LLM interface for testing.

**Parameters:**

- **`args`**: `Any`
- **`kwargs`**: `Any`

---

### `OpenAIAdapter`

**Module:** `hexdag.stdlib.adapters.openai.OpenAIAdapter`

Unified OpenAI implementation of the LLM port.

**Parameters:**

- **`args`**: `Any`
- **`kwargs`**: `Any`

---

### `PgVectorAdapter`

**Module:** `hexdag.stdlib.adapters.database.pgvector.PgVectorAdapter`

PostgreSQL adapter with pgvector extension support.

**Parameters:**

- **`args`**: `Any`
- **`kwargs`**: `Any`

---

### `SQLAlchemyAdapter`

**Module:** `hexdag.stdlib.adapters.database.sqlalchemy.SQLAlchemyAdapter`

*Could not load component: Cannot resolve 'hexdag.stdlib.adapters.database.sqlalchemy.SQLAlchemyAdapter': Class 'SQLAlchemyAdapter' not found in 'hexdag.stdlib.adapters.database.sqlalchemy'. Available: *

---

### `SQLiteAdapter`

**Module:** `hexdag.stdlib.adapters.database.sqlite.SQLiteAdapter`

Async SQLite adapter for database port.

**Parameters:**

- **`args`**: `Any`
- **`kwargs`**: `Any`

---

### `SQLiteMemoryAdapter`

**Module:** `hexdag.stdlib.adapters.memory.SQLiteMemoryAdapter`

Memory adapter backed by SQLite database.

**Parameters:**

- **`args`**: `Any`
- **`kwargs`**: `Any`

---

### `anthropic_adapter`

**Module:** `hexdag.stdlib.adapters.anthropic.AnthropicAdapter`

Anthropic implementation of the LLM port.

**Parameters:**

- **`args`**: `Any`
- **`kwargs`**: `Any`

---

### `csv_adapter`

**Module:** `hexdag.stdlib.adapters.database.csv.CsvAdapter`

*Could not load component: Cannot resolve 'hexdag.stdlib.adapters.database.csv.CsvAdapter': Class 'CsvAdapter' not found in 'hexdag.stdlib.adapters.database.csv'. Available: *

---

### `database:csv`

**Module:** `hexdag.stdlib.adapters.database.csv.CsvAdapter`

*Could not load component: Cannot resolve 'hexdag.stdlib.adapters.database.csv.CsvAdapter': Class 'CsvAdapter' not found in 'hexdag.stdlib.adapters.database.csv'. Available: *

---

### `database:mock`

**Module:** `hexdag.stdlib.adapters.mock.MockDatabaseAdapter`

Mock implementation of Database for testing and demos.

**Parameters:**

- **`args`**: `Any`
- **`kwargs`**: `Any`

---

### `database:pgvector`

**Module:** `hexdag.stdlib.adapters.database.pgvector.PgVectorAdapter`

PostgreSQL adapter with pgvector extension support.

**Parameters:**

- **`args`**: `Any`
- **`kwargs`**: `Any`

---

### `database:sqlalchemy`

**Module:** `hexdag.stdlib.adapters.database.sqlalchemy.SQLAlchemyAdapter`

*Could not load component: Cannot resolve 'hexdag.stdlib.adapters.database.sqlalchemy.SQLAlchemyAdapter': Class 'SQLAlchemyAdapter' not found in 'hexdag.stdlib.adapters.database.sqlalchemy'. Available: *

---

### `database:sqlite`

**Module:** `hexdag.stdlib.adapters.database.sqlite.SQLiteAdapter`

Async SQLite adapter for database port.

**Parameters:**

- **`args`**: `Any`
- **`kwargs`**: `Any`

---

### `embedding:mock`

**Module:** `hexdag.stdlib.adapters.mock.MockEmbedding`

Mock implementation of the Embedding interface for testing.

**Parameters:**

- **`args`**: `Any`
- **`kwargs`**: `Any`

---

### `file_memory_adapter`

**Module:** `hexdag.stdlib.adapters.memory.FileMemoryAdapter`

Memory adapter backed by file system.

**Parameters:**

- **`args`**: `Any`
- **`kwargs`**: `Any`

---

### `in_memory_memory`

**Module:** `hexdag.stdlib.adapters.memory.InMemoryMemory`

In-memory implementation of Memory for testing.

**Parameters:**

- **`args`**: `Any`
- **`kwargs`**: `Any`

---

### `llm:anthropic`

**Module:** `hexdag.stdlib.adapters.anthropic.AnthropicAdapter`

Anthropic implementation of the LLM port.

**Parameters:**

- **`args`**: `Any`
- **`kwargs`**: `Any`

---

### `llm:mock`

**Module:** `hexdag.stdlib.adapters.mock.MockLLM`

Mock implementation of the LLM interface for testing.

**Parameters:**

- **`args`**: `Any`
- **`kwargs`**: `Any`

---

### `llm:openai`

**Module:** `hexdag.stdlib.adapters.openai.OpenAIAdapter`

Unified OpenAI implementation of the LLM port.

**Parameters:**

- **`args`**: `Any`
- **`kwargs`**: `Any`

---

### `local_secret_adapter`

**Module:** `hexdag.stdlib.adapters.secret.LocalSecretAdapter`

Local secret adapter that reads from environment variables.

**Parameters:**

- **`args`**: `Any`
- **`kwargs`**: `Any`

---

### `memory:file`

**Module:** `hexdag.stdlib.adapters.memory.FileMemoryAdapter`

Memory adapter backed by file system.

**Parameters:**

- **`args`**: `Any`
- **`kwargs`**: `Any`

---

### `memory:in_memory`

**Module:** `hexdag.stdlib.adapters.memory.InMemoryMemory`

In-memory implementation of Memory for testing.

**Parameters:**

- **`args`**: `Any`
- **`kwargs`**: `Any`

---

### `memory:sqlite`

**Module:** `hexdag.stdlib.adapters.memory.SQLiteMemoryAdapter`

Memory adapter backed by SQLite database.

**Parameters:**

- **`args`**: `Any`
- **`kwargs`**: `Any`

---

### `mock_database_adapter`

**Module:** `hexdag.stdlib.adapters.mock.MockDatabaseAdapter`

Mock implementation of Database for testing and demos.

**Parameters:**

- **`args`**: `Any`
- **`kwargs`**: `Any`

---

### `mock_embedding`

**Module:** `hexdag.stdlib.adapters.mock.MockEmbedding`

Mock implementation of the Embedding interface for testing.

**Parameters:**

- **`args`**: `Any`
- **`kwargs`**: `Any`

---

### `mock_llm`

**Module:** `hexdag.stdlib.adapters.mock.MockLLM`

Mock implementation of the LLM interface for testing.

**Parameters:**

- **`args`**: `Any`
- **`kwargs`**: `Any`

---

### `open_ai_adapter`

**Module:** `hexdag.stdlib.adapters.openai.OpenAIAdapter`

Unified OpenAI implementation of the LLM port.

**Parameters:**

- **`args`**: `Any`
- **`kwargs`**: `Any`

---

### `pg_vector_adapter`

**Module:** `hexdag.stdlib.adapters.database.pgvector.PgVectorAdapter`

PostgreSQL adapter with pgvector extension support.

**Parameters:**

- **`args`**: `Any`
- **`kwargs`**: `Any`

---

### `secret:local`

**Module:** `hexdag.stdlib.adapters.secret.LocalSecretAdapter`

Local secret adapter that reads from environment variables.

**Parameters:**

- **`args`**: `Any`
- **`kwargs`**: `Any`

---

### `sq_lite_adapter`

**Module:** `hexdag.stdlib.adapters.database.sqlite.SQLiteAdapter`

Async SQLite adapter for database port.

**Parameters:**

- **`args`**: `Any`
- **`kwargs`**: `Any`

---

### `sq_lite_memory_adapter`

**Module:** `hexdag.stdlib.adapters.memory.SQLiteMemoryAdapter`

Memory adapter backed by SQLite database.

**Parameters:**

- **`args`**: `Any`
- **`kwargs`**: `Any`

---

### `sql_alchemy_adapter`

**Module:** `hexdag.stdlib.adapters.database.sqlalchemy.SQLAlchemyAdapter`

*Could not load component: Cannot resolve 'hexdag.stdlib.adapters.database.sqlalchemy.SQLAlchemyAdapter': Class 'SQLAlchemyAdapter' not found in 'hexdag.stdlib.adapters.database.sqlalchemy'. Available: *

---
