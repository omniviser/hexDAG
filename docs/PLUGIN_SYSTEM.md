# HexDAG Plugin System Documentation

## Overview

HexDAG uses a powerful plugin architecture that allows you to extend the framework with custom adapters and components. The system is built on the Hexagonal Architecture pattern (Ports and Adapters), providing clean separation between business logic and infrastructure.

## Quick Start

Get started with plugin development in minutes:

```bash
# Create a new plugin
hexdag plugin new redis_adapter --port memory

# Develop and test
hexdag plugin lint redis_adapter
hexdag plugin test redis_adapter

# Install the plugin
hexdag plugin install redis_adapter
```

### Plugin Structure

```
hexdag_plugins/my_adapter/
├── my_adapter.py        # Implement port interface
├── pyproject.toml       # Dependencies
└── tests/               # Plugin tests
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `new <name>` | Create plugin from template |
| `list` | Show all installed plugins |
| `lint <name>` | Lint code with auto-fix |
| `format <name>` | Format code |
| `test <name>` | Run plugin tests |
| `install <name>` | Install plugin dependencies |
| `check-all` | Check all plugins |

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Plugin Loading Process](#plugin-loading-process)
3. [Configuration](#configuration)
4. [Creating Custom Plugins](#creating-custom-plugins)
5. [Plugin Organization](#plugin-organization)
6. [Best Practices](#best-practices)
7. [Examples](#examples)

## Core Concepts

### Ports and Adapters Pattern

The plugin system is based on two key concepts:

- **Ports**: Abstract interfaces that define contracts for external services
- **Adapters**: Concrete implementations of ports that integrate with specific services

```python
# Port definition (interface) - defined in hexdag.core.ports
from hexdag.core.ports.llm import LLM

class LLM(Protocol):
    """Abstract interface for language models."""
    async def aresponse(self, messages: list[Message]) -> str: ...

# Adapter implementation - implements the port interface
class OpenAILLM(LLM):
    """OpenAI implementation of the LLM port."""
    async def aresponse(self, messages: list[Message]) -> str:
        # Actual OpenAI API call
        return response
```

### Component Resolution

Components are resolved by their full module path:

```python
from hexdag.core.resolver import resolve

# Resolve a component by module path
OpenAILLM = resolve("mycompany.adapters.OpenAILLM")

# Use in YAML pipelines
# adapter: mycompany.adapters.OpenAILLM
```

## Plugin Loading Process

### 1. Two-Phase Loading

HexDAG uses a two-phase loading process to ensure proper initialization:

```
Phase 1: Load Ports (interfaces)
   ↓
Phase 2: Load Adapters (implementations)
```

This ensures that all interfaces are available before their implementations are registered.

### 2. Loading Sequence

```python
# 1. Bootstrap reads configuration
bootstrap_registry("hexdag.toml")

# 2. Load modules (ports first)
for module in config.modules:
    load_module(module)  # e.g., "hexdag.core.ports"

# 3. Load plugins (adapters)
for plugin in config.plugins:
    load_plugin(plugin)  # e.g., "hexdag.adapters.openai"

# 4. Components are now available in registry
```

### 3. Namespace Management

Components are organized in namespaces:

- `core`: Framework built-in components
- `plugin`: User and third-party plugins
- `dev`: Development-only components (when `dev_mode=true`)

## Configuration

### Main Configuration (`pyproject.toml`)

```toml
[tool.hexdag]
# Core modules to load (ports and core components)
modules = [
    "hexdag.core.ports",               # Port definitions
    "hexdag.builtin.nodes",   # Core node types
]

# Default plugins to load
plugins = [
    "hexdag.builtin.adapters.local",  # Local in-process adapters
]

# Enable development mode for testing
dev_mode = true

[tool.hexdag.settings]
# Global settings
log_level = "INFO"
enable_metrics = true
```

### Plugin-Specific Configuration

Plugins can have their own configuration files:

```toml
# hexdag/adapters/openai/hexdag.toml
modules = [
    "hexdag.core.ports",  # Required ports
]

plugins = [
    "hexdag.adapters.openai",  # The plugin module
]

[settings.openai]
api_key_env = "OPENAI_API_KEY"
model = "gpt-4"
temperature = 0.7
max_tokens = 2000
```

### Loading Custom Configurations

HexDAG loads plugins through the standard Python import mechanism. Configure plugins
in your `pyproject.toml` or directly import them:

```python
# Import plugins directly by their module path
from hexdag_plugins.redis_adapter import RedisAdapter

# Or use the resolver to load by path string
from hexdag.core.resolver import resolve

RedisAdapter = resolve("hexdag_plugins.redis_adapter.RedisAdapter")
```

For environment-specific configuration, use environment variables in your YAML:

```yaml
spec:
  ports:
    llm:
      adapter: mycompany.adapters.CustomLLM
      config:
        api_key: ${LLM_API_KEY}  # Resolved from environment
```

## Creating Custom Plugins

### Step 1: Define Your Adapter

```python
# my_company/adapters/custom_llm.py
from hexdag.core.ports.llm import LLM
import os

class CustomLLM(LLM):
    """Custom LLM implementation."""

    def __init__(
        self,
        api_url: str = "https://api.example.com",
        api_key: str | None = None,
        timeout: float = 30.0
    ):
        self.api_url = api_url
        self.api_key = api_key or os.environ.get("CUSTOM_LLM_API_KEY")
        self.timeout = timeout

    async def aresponse(self, messages: list[dict]) -> str:
        # Your implementation here
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.api_url,
                json={"messages": messages},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout
            )
            return response.json()["content"]
```

### Step 2: Create Plugin Module Structure

```
my_company/
├── __init__.py
├── adapters/
│   ├── __init__.py
│   ├── custom_llm.py
│   ├── custom_database.py
│   └── custom_memory.py
├── hexdag.toml          # Plugin configuration
└── README.md
```

### Step 3: Export Components

```python
# my_company/adapters/__init__.py
from my_company.adapters.custom_llm import CustomLLM
from my_company.adapters.custom_database import CustomDatabase
from my_company.adapters.custom_memory import CustomMemory

__all__ = [
    "CustomLLM",
    "CustomDatabase",
    "CustomMemory",
]
```

### Step 4: Create Configuration

```toml
# my_company/hexdag.toml
modules = [
    "hexdag.core.ports",  # Required for port definitions
]

plugins = [
    "my_company.adapters",  # Your plugin module
]

[settings.custom_llm]
api_url = "https://api.mycompany.com/v1/chat"
api_key_env = "MY_COMPANY_API_KEY"  # Read from environment
timeout = 60.0

[settings.custom_database]
connection_string_env = "DATABASE_URL"
pool_size = 10
```

### Step 5: Use Your Plugin

Reference your adapter by its full module path in YAML:

```yaml
# pipeline.yaml
apiVersion: hexdag/v1
kind: Pipeline
spec:
  ports:
    llm:
      adapter: my_company.adapters.CustomLLM
      config:
        api_url: https://api.mycompany.com/v1/chat
        timeout: 60.0
```

Or instantiate directly in Python:

```python
from my_company.adapters import CustomLLM

llm = CustomLLM(api_url="https://api.mycompany.com/v1/chat")
response = await llm.aresponse(messages)
```

## Plugin Organization

### Directory Structure Best Practices

```
hexdag/
├── adapters/
│   ├── local/           # In-process implementations
│   │   ├── __init__.py
│   │   ├── in_memory_memory.py
│   │   ├── hexdag.toml
│   │   └── README.md
│   │
│   ├── mock/            # Testing doubles
│   │   ├── __init__.py
│   │   ├── mock_llm.py
│   │   ├── mock_database.py
│   │   ├── hexdag.toml
│   │   └── README.md
│   │
│   ├── openai/          # OpenAI integration
│   │   ├── __init__.py
│   │   ├── openai_llm.py
│   │   ├── hexdag.toml
│   │   └── README.md
│   │
│   └── configs.py       # Shared configuration models
```

### Plugin Categories

1. **Local Adapters**: In-process implementations with no external dependencies
   - Example: `InMemoryMemory`, `FileSystemStorage`
   - Use case: Development, testing, simple deployments

2. **Mock Adapters**: Test doubles for development and testing
   - Example: `MockLLM`, `MockDatabase`
   - Use case: Unit tests, CI/CD, demos

3. **External Service Adapters**: Integrate with external APIs and services
   - Example: `OpenAILLM`, `PostgreSQLDatabase`
   - Use case: Production deployments

4. **Custom Business Logic**: Company-specific implementations
   - Example: `ProprietaryLLM`, `InternalDataStore`
   - Use case: Enterprise deployments

## Best Practices

### 1. Configuration Management

```python
# Use Pydantic for type-safe configuration
from pydantic import BaseModel, Field, SecretStr

class MyAdapterConfig(BaseModel):
    """Configuration with validation."""

    # Required fields
    api_url: str = Field(description="API endpoint")

    # Optional with defaults
    timeout: float = Field(default=30.0, ge=0)

    # Sensitive data
    api_key: SecretStr = Field(description="API key")

    # Environment variable support
    @classmethod
    def from_env(cls) -> "MyAdapterConfig":
        import os
        return cls(
            api_url=os.getenv("MY_API_URL", "https://default.api.com"),
            api_key=os.environ["MY_API_KEY"],  # Required
            timeout=float(os.getenv("MY_TIMEOUT", "30"))
        )
```

### 2. Error Handling

```python
class RobustAdapter(LLM):
    """Adapter with proper error handling."""

    async def aresponse(self, messages: list[dict]) -> str:
        try:
            # Attempt operation
            return await self._call_api(messages)
        except httpx.TimeoutException:
            # Handle specific errors
            raise TimeoutError("API request timed out")
        except httpx.HTTPStatusError as e:
            # Provide context
            raise RuntimeError(f"API error: {e.response.status_code}")
        except Exception as e:
            # Log and re-raise
            logger.error(f"Unexpected error: {e}")
            raise
```

### 3. Testing Your Plugin

```python
# tests/test_custom_adapter.py
import pytest
from my_company.adapters import CustomLLM
from my_company.adapters.configs import CustomLLMConfig

@pytest.fixture
def adapter():
    """Create adapter with test configuration."""
    config = CustomLLMConfig(
        api_url="https://test.api.com",
        api_key="test-key",
        timeout=5.0
    )
    return CustomLLM(config)

@pytest.mark.asyncio
async def test_adapter_response(adapter, mocker):
    """Test adapter response."""
    # Mock external call
    mock_post = mocker.patch("httpx.AsyncClient.post")
    mock_post.return_value.json.return_value = {"content": "Test response"}

    # Test
    messages = [Message(role="user", content="Hello")]
    response = await adapter.aresponse(messages)

    # Verify
    assert response == "Test response"
    mock_post.assert_called_once()
```

### 4. Documentation

Always include a README with your plugin:

```markdown
# Custom LLM Adapter

## Installation

```bash
pip install my-company-hexdag-plugin
```

## Configuration

Add to your `hexdag.toml`:

```toml
plugins = ["my_company.adapters"]

[settings.custom_llm]
api_url = "https://api.example.com"
api_key_env = "CUSTOM_API_KEY"
```

## Usage

```yaml
# In YAML pipeline
spec:
  ports:
    llm:
      adapter: my_company.adapters.CustomLLM
```
```

## Examples

### Example 1: Simple Memory Adapter

```python
# simple_memory.py
from hexdag.core.ports.memory import Memory
from typing import Any

class SimpleMemory(Memory):
    """Simple in-memory storage."""

    def __init__(self):
        self.data = {}

    async def aget(self, key: str) -> Any:
        return self.data.get(key)

    async def aset(self, key: str, value: Any) -> None:
        self.data[key] = value

    async def adelete(self, key: str) -> None:
        self.data.pop(key, None)
```

Use in YAML: `adapter: my_company.adapters.SimpleMemory`

### Example 2: Database Adapter with Connection Pooling

```python
# postgres_adapter.py
import asyncpg
from hexdag.core.ports.database import Database

class PostgreSQLAdapter(Database):
    """PostgreSQL database adapter."""

    def __init__(self, connection_string: str, pool_size: int = 10):
        self.connection_string = connection_string
        self.pool_size = pool_size
        self.pool = None

    async def connect(self):
        """Initialize connection pool."""
        self.pool = await asyncpg.create_pool(
            self.config.connection_string,
            min_size=self.config.min_connections,
            max_size=self.config.max_connections
        )

    async def aexecute_query(self, query: str, params: dict) -> list[dict]:
        """Execute query with connection from pool."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params.values())
            return [dict(row) for row in rows]

    async def close(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
```

### Example 3: Multi-Provider Adapter

```python
# multi_llm_adapter.py
from hexdag.core.ports.llm import LLM
from hexdag.core.resolver import resolve

class MultiProviderLLM(LLM):
    """Adapter that can switch between multiple LLM providers."""

    def __init__(
        self,
        providers: dict[str, str],  # name -> module path
        default_provider: str = "openai"
    ):
        self.provider_instances = {}

        # Instantiate configured providers
        for name, module_path in providers.items():
            provider_cls = resolve(module_path)
            self.provider_instances[name] = provider_cls()

        self.current_provider = default_provider

    async def aresponse(self, messages: list[dict]) -> str:
        """Route to current provider."""
        provider = self.provider_instances[self.current_provider]
        return await provider.aresponse(messages)

    def switch_provider(self, provider_name: str):
        """Switch active provider."""
        if provider_name not in self.provider_instances:
            raise ValueError(f"Unknown provider: {provider_name}")
        self.current_provider = provider_name
```

### Example 4: Environment-Specific Configuration

Use different YAML pipeline files for different environments:

```yaml
# config/production/pipeline.yaml
spec:
  ports:
    llm:
      adapter: hexdag.builtin.adapters.openai.OpenAIAdapter
      config:
        model: gpt-4

# config/testing/pipeline.yaml
spec:
  ports:
    llm:
      adapter: hexdag.builtin.adapters.mock.MockLLM
```

```python
# main.py
import os
from hexdag import YamlPipelineBuilder

environment = os.getenv("ENVIRONMENT", "development")
pipeline_path = f"config/{environment}/pipeline.yaml"

builder = YamlPipelineBuilder()
graph, config = builder.build_from_yaml_file(pipeline_path)
```

## Advanced Topics

### Dynamic Plugin Discovery

```python
# plugin_discovery.py
from pathlib import Path
import importlib.util

def discover_plugins(plugins_dir: Path) -> list[str]:
    """Discover all plugins in a directory."""
    plugins = []

    for plugin_path in plugins_dir.glob("*/hexdag.toml"):
        plugin_name = plugin_path.parent.name
        plugins.append(f"custom_plugins.{plugin_name}")

    return plugins

# Use discovered plugins
plugins = discover_plugins(Path("./custom_plugins"))
for plugin in plugins:
    bootstrap_registry(f"custom_plugins/{plugin}/hexdag.toml")
```

### Plugin Versioning

```python
# versioned_adapter.py
from packaging import version
from hexdag.core.ports.llm import LLM

class VersionedLLM(LLM):
    """Adapter with version information."""

    VERSION = "2.0.0"
    MIN_HEXDAG_VERSION = "1.0.0"

    def __init__(self):
        # Check compatibility
        import hexdag
        if version.parse(hexdag.__version__) < version.parse(self.MIN_HEXDAG_VERSION):
            raise RuntimeError(f"HexDAG {hexdag.__version__} is too old for this adapter")
```

### Plugin Lifecycle Hooks

```python
# lifecycle_adapter.py
class LifecycleLLM(LLM):
    """Adapter with lifecycle management."""

    async def asetup(self):
        """Called when adapter is first created."""
        await self._connect_to_service()
        await self._warm_up_cache()

    async def aclose(self):
        """Called when adapter is being destroyed."""
        await self._disconnect()
        await self._cleanup_resources()

    async def health_check(self) -> bool:
        """Check if adapter is healthy."""
        try:
            await self._ping_service()
            return True
        except Exception:
            return False
```

## Troubleshooting

### Common Issues

1. **Plugin not found**
   - Check that the full module path is correct in your YAML configuration
   - Verify the adapter class implements the port interface
   - Ensure the module is on the Python path

2. **Port not implemented error**
   - Verify the adapter class inherits from the port interface
   - Check that all required methods are implemented
   - Ensure async methods use `async def`

3. **Configuration not loading**
   - Check the configuration file path is correct
   - Verify TOML syntax is valid
   - Ensure environment variables are set if using `_env` suffix

4. **Import errors**
   - Make sure the plugin module is on the Python path
   - Check for circular imports
   - Verify all dependencies are installed

### Debug Mode

Enable debug logging to troubleshoot plugin loading:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# The resolver logs detailed information when resolving component paths
from hexdag.core.resolver import resolve
adapter = resolve("mycompany.adapters.CustomAdapter")  # Logs resolution details
```

## Summary

The HexDAG plugin system provides:

- **Clean Architecture**: Separation of business logic and infrastructure
- **Type Safety**: Pydantic models and type hints throughout
- **Flexibility**: Multiple configuration options and loading strategies
- **Testability**: Easy to mock and test with dependency injection
- **Extensibility**: Simple to add new adapters and ports

For more examples, see the `examples/` directory in the HexDAG repository.
