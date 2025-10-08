# HexDAG MySQL Adapter

A production-ready MySQL database adapter plugin for the hexDAG framework.

## Features

- 🚀 **Production-Ready**: Built for high-performance production deployments
- 📦 **JSON Document Storage**: Leverage MySQL's native JSON support
- 🔄 **Async Operations**: Full async/await support for all operations
- 🔒 **Transaction Support**: ACID compliance with proper transaction handling
- 🎯 **Type-Safe**: Full type hints and Pydantic validation
- 🔌 **Plugin Architecture**: Seamlessly integrates with hexDAG's plugin system

## Installation

### Local Development

For local development within the hexDAG project:

```bash
# Using uv (recommended for hexDAG development)
uv pip install -e hexdag_plugins/mysql_adapter/

# Or using pip
uv pip install -e hexdag_plugins/mysql_adapter/
```

### With development dependencies

```bash
pip install -e "hexdag_plugins/mysql_adapter/[dev]"
```

## Configuration

### Environment Variables

Set these environment variables for default connection:

```bash
export MYSQL_HOST=localhost
export MYSQL_PORT=3306
export MYSQL_USER=your_user
export MYSQL_PASSWORD=your_password
export MYSQL_DATABASE=hexdag
```

### Programmatic Configuration

```python
from hexdag_plugins.mysql_adapter import MySQLAdapter

adapter = MySQLAdapter(
    host="your-mysql-host",
    port=3306,
    user="your-user",
    password="your-password",
    database="hexdag",
    charset="utf8mb4"
)
```

## Usage

### Basic Operations

```python
import asyncio
from hexdag_plugins.mysql_adapter import MySQLAdapter

async def main():
    # Initialize adapter
    adapter = MySQLAdapter(
        host="localhost",
        user="root",
        password="password",
        database="hexdag"
    )

    # Insert document
    doc_id = await adapter.ainsert("users", {
        "name": "John Doe",
        "email": "john@example.com",
        "metadata": {
            "role": "admin",
            "permissions": ["read", "write", "delete"]
        }
    })

    # Query documents
    users = await adapter.aquery(
        "users",
        filter={"metadata.role": "admin"}
    )

    # Update document
    await adapter.aupdate("users", doc_id, {
        "last_login": "2024-01-01T12:00:00Z"
    })

    # Delete document
    await adapter.adelete("users", doc_id)

asyncio.run(main())
```

### Integration with hexDAG

The adapter automatically registers with hexDAG's registry system:

```python
from hexai.core.bootstrap import bootstrap_registry
from hexai.core.registry import registry

# Bootstrap with MySQL plugin
bootstrap_registry()

# Get MySQL adapter from registry
mysql_info = registry.get_info("mysql", namespace="plugin")
adapter = mysql_info.get_instance(
    host="localhost",
    user="root",
    password="password"
)
```

## Database Schema

The adapter creates a flexible document store schema:

```sql
CREATE TABLE hexdag_documents (
    id VARCHAR(255) PRIMARY KEY,
    collection VARCHAR(255) NOT NULL,
    document JSON NOT NULL,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_collection (collection),
    INDEX idx_created (created_at),
    INDEX idx_updated (updated_at)
);
```

## Advanced Features

### JSON Path Queries

Leverage MySQL's JSON functions for complex queries:

```python
# Query nested JSON fields
results = await adapter.aquery(
    "products",
    filter={
        "specs.memory": "16GB",
        "price": {"$lt": 1000}
    }
)
```

### Concurrent Operations

The adapter supports concurrent operations safely:

```python
import asyncio

async def bulk_insert(adapter, collection, documents):
    tasks = [
        adapter.ainsert(collection, doc)
        for doc in documents
    ]
    return await asyncio.gather(*tasks)
```

### Connection Pooling

The adapter uses PyMySQL with connection management for optimal performance.

## Testing

### Run Tests

```bash
pytest tests/
```

### Test with Docker MySQL

```bash
# Start MySQL container
docker run -d \
    --name mysql-test \
    -e MYSQL_ROOT_PASSWORD=test \
    -e MYSQL_DATABASE=hexdag_test \
    -p 3306:3306 \
    mysql:8.0

# Run tests
MYSQL_TEST_HOST=localhost \
MYSQL_TEST_USER=root \
MYSQL_TEST_PASSWORD=test \
MYSQL_TEST_DATABASE=hexdag_test \
pytest tests/
```

## Requirements

- Python 3.12+
- MySQL 5.7+ (8.0+ recommended for full JSON support)
- PyMySQL 1.1.0+

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please use the GitHub issue tracker.
