# hexdag-storage

Low-level storage infrastructure for hexDAG - SQL databases, vector stores, and file storage.

## What This Package Provides

- **SQL Adapters**: PostgreSQL, MySQL, SQLite (with connection pooling)
- **Vector Stores**: pgvector, ChromaDB, in-memory
- **File Storage**: Local filesystem (S3, Azure, GCS planned)
- **Production-Ready**: Connection pooling, health checks, async-first

## What This Is NOT

- ❌ RAG business logic (no document processing, chunking, embeddings)
- ❌ AI/LLM integration
- ❌ High-level workflows or pipelines
- ❌ Domain-specific operations

## Installation

```bash
# Install with all storage backends
uv pip install -e 'hexdag_plugins/storage[all]'

# Or install specific backends:
uv pip install -e 'hexdag_plugins/storage[postgresql]'  # PostgreSQL + pgvector
uv pip install -e 'hexdag_plugins/storage[mysql]'       # MySQL
uv pip install -e 'hexdag_plugins/storage[chromadb]'    # ChromaDB
uv pip install -e 'hexdag_plugins/storage[pgvector]'    # pgvector only

# Basic installation (no optional dependencies)
uv pip install -e hexdag_plugins/storage
```

**Note:** Quotes around the package path with extras are required for zsh to prevent glob expansion.

## Usage

### Vector Stores

```python
from hexdag.core.registry import registry
from hexdag.core.bootstrap import bootstrap_registry

bootstrap_registry()

# In-memory (for testing)
vector_store = registry.get("in_memory_vector", namespace="storage")()
await vector_store.aadd_documents(
    documents=[{"text": "Python"}],
    embeddings=[[0.1, 0.2, 0.3]]
)
results = await vector_store.asearch("", query_embedding=[0.1, 0.2, 0.3])

# PostgreSQL + pgvector (production)
pgvector = registry.get("pgvector", namespace="storage")(
    connection_string="postgresql+asyncpg://localhost/mydb",
    pool_size=10,
    table_name="embeddings",
    embedding_dim=384
)
await pgvector.asetup()
await pgvector.aadd_documents(documents, embeddings)
```

### File Storage

```python
# Local file storage
storage = registry.get("local", namespace="storage")(
    base_path="./data"
)

# Upload file
await storage.aupload("document.pdf", "docs/document.pdf")

# List files
files = await storage.alist(prefix="docs/")

# Download file
await storage.adownload("docs/document.pdf", "/tmp/document.pdf")

# Check exists
exists = await storage.aexists("docs/document.pdf")

# Get metadata
metadata = await storage.aget_metadata("docs/document.pdf")

# Delete file
await storage.adelete("docs/document.pdf")

# Health check
health = await storage.ahealth_check()
```

## Architecture

### Ports

**In hexDAG Core:**
- `DatabasePort` - SQL database operations
- `FileStoragePort` - File storage operations

**In hexdag-storage:**
- `VectorStorePort` - Specialized for vector similarity search

### Implementations

```
hexdag_plugins/storage/
├── sql/              # DatabasePort implementations
│   ├── postgresql.py
│   ├── mysql.py
│   └── sqlite.py
│
├── vector/           # VectorStorePort implementations
│   ├── pgvector.py   # PostgreSQL + pgvector
│   ├── chromadb.py   # ChromaDB
│   └── in_memory.py  # In-memory (testing)
│
└── file/             # FileStoragePort implementations
    └── local.py      # Local filesystem
```

## Benefits

### SQLAlchemy Connection Pooling

All SQL-based adapters use SQLAlchemy's async engine with:
- Connection pooling (5-20 connections)
- Automatic health checks (`pool_pre_ping`)
- Connection recycling
- Production-ready performance

### Consistent Configuration

All adapters use:
- `AdapterConfig` + `SecretField` for configuration
- Environment variable support
- Type validation with Pydantic

### Health Monitoring

All adapters implement `ahealth_check()`:
- Test connectivity
- Check pool status
- Measure latency
- Return structured `HealthStatus`

## Development

```bash
# Run tests
uv run pytest tests/

# Linting
uv run ruff check .

# Type checking
uv run pyright .
```

## Future Plans

- S3 file storage adapter
- Azure Blob storage adapter
- Google Cloud Storage adapter
- Redis vector store adapter

## License

MIT
