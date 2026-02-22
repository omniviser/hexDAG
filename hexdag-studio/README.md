# hexdag-studio

Visual Studio UI for hexDAG pipelines.

## Installation

```bash
pip install hexdag-studio
```

Or with uv:

```bash
uv pip install hexdag-studio
```

## Usage

Start the studio with a workspace directory:

```bash
hexdag-studio ./pipelines/
```

### Basic Options

```bash
hexdag-studio ./pipelines/ --port 8080       # Custom port
hexdag-studio ./pipelines/ --host 0.0.0.0    # Bind to all interfaces
hexdag-studio ./pipelines/ --no-browser      # Don't auto-open browser
```

### Loading Plugins

Load custom plugins to extend Studio with additional nodes and adapters:

```bash
# Load a single plugin
hexdag-studio ./pipelines/ --plugin ./hexdag_plugins/azure

# Load multiple plugins
hexdag-studio ./pipelines/ --plugin ./hexdag_plugins/azure --plugin ./hexdag_plugins/etl

# Load all plugins in a directory (each subdirectory is treated as a plugin)
hexdag-studio ./pipelines/ --plugin ./hexdag_plugins --with-subdirs

# Auto-install plugin dependencies before loading
hexdag-studio ./pipelines/ --plugin ./hexdag_plugins/azure --install-plugin-deps
```

### CLI Reference

| Option | Description |
|--------|-------------|
| `PATH` | Workspace directory containing pipeline YAML files (required) |
| `--host, -h` | Host to bind to (default: `127.0.0.1`) |
| `--port, -p` | Port to bind to (default: `3141`) |
| `--no-browser` | Don't open browser automatically |
| `--plugin` | Plugin directory path (can be used multiple times) |
| `--with-subdirs` | Treat each subdirectory of `--plugin` as a separate plugin |
| `--install-plugin-deps` | Install plugin dependencies from `pyproject.toml` before loading |

## Development

### Backend (Python)

```bash
cd hexdag-studio
uv pip install -e .
```

### Frontend (React)

```bash
cd hexdag-studio/ui
npm install
npm run dev    # Development server with hot reload
npm run build  # Production build
```

## Features

- Visual DAG editor with drag-and-drop nodes
- YAML editor with syntax highlighting
- Real-time validation
- Pipeline execution with mock adapters
- Port configuration for LLM, memory, database adapters
- Plugin support for custom nodes and adapters

## Architecture

```
hexdag-studio/
├── hexdag_studio/           # Python backend
│   ├── cli.py              # CLI entry point
│   └── server/
│       ├── main.py         # FastAPI app
│       └── routes/         # API endpoints
├── ui/                      # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   └── lib/            # Utilities and state
│   └── dist/               # Built static files
└── examples/               # Example pipelines for testing
```

## Component Discovery

hexdag-studio uses a **unified component discovery system** that provides a single source of truth for all nodes, adapters, tools, and macros. This eliminates duplicates and ensures consistency across the UI.

### How It Works

All component discovery is handled by `hexdag.api.components`, which the Studio delegates to. Components are uniquely identified by their **module path** (the Python import path).

```
module_path examples:
├── hexdag.stdlib.nodes.LLMNode          → Built-in (core)
├── hexdag.stdlib.adapters.mock.MockLLM  → Built-in (core)
├── hexdag_plugins.mysql_adapter.MySQLAdapter → Plugin: mysql_adapter
├── hexdag_plugins.azure.AzureOpenAI      → Plugin: azure
└── mycompany.adapters.CustomLLM          → User plugin: mycompany
```

### Component Identification

The system determines where a component belongs based on its `module_path`:

| Module Path Pattern | Classification |
|---------------------|----------------|
| `hexdag.stdlib.*` | Built-in (core) - shown in "Core Nodes" |
| `hexdag_plugins.<name>.*` | Official plugin - grouped by `<name>` |
| `<package>.*` | User plugin - grouped by top-level package |

### API Endpoints

| Endpoint | Returns |
|----------|---------|
| `/api/registry/nodes` | Built-in nodes only (from `hexdag.stdlib.*`) |
| `/api/registry/adapters` | All adapters with `plugin` field |
| `/api/plugins` | Plugin metadata grouped by module path |
| `/api/plugins/nodes/all` | Plugin nodes only (excludes built-in) |
| `/api/plugins/adapters/all` | All adapters with `plugin` field |

### No Duplicates

Since `module_path` is the unique identifier:
- Each component appears exactly once
- Built-in nodes appear in "Core Nodes" section
- Plugin nodes appear under their respective plugin
- No overlap between `/registry` and `/plugins` endpoints

### Using in YAML

Reference components by their module path:

```yaml
apiVersion: hexdag/v1
kind: Pipeline
spec:
  ports:
    llm:
      adapter: hexdag_plugins.azure.AzureOpenAI  # Plugin adapter
      config:
        model: gpt-4
    database:
      adapter: mycompany.adapters.PostgresAdapter  # User adapter
  nodes:
    - kind: llm_node  # Built-in (alias)
      metadata:
        name: analyzer
    - kind: hexdag_plugins.etl.TransformNode  # Plugin node
      metadata:
        name: transformer
```

## Plugin System

hexdag-studio discovers plugins from the `hexdag_plugins/` directory or custom paths specified via `--plugin`. Plugins can provide custom **nodes** and **adapters**.

### Plugin Discovery

By default, Studio scans the `hexdag_plugins/` directory. When using `--plugin`, only the specified plugins are loaded:

```bash
# Only loads azure plugin (not other plugins in hexdag_plugins/)
hexdag-studio ./pipelines/ --plugin ./hexdag_plugins/azure
```

### Plugin Structure

```
my_plugin/
├── __init__.py      # Exports adapters/nodes via __all__
├── pyproject.toml   # Plugin dependencies (optional)
├── adapters.py      # Custom adapters
└── nodes.py         # Custom nodes
```

### Plugin Dependencies

Plugins can have their own dependencies defined in `pyproject.toml`:

```toml
# my_plugin/pyproject.toml
[project]
name = "hexdag-plugin-azure"
version = "0.2.0"
dependencies = [
    "azure-identity>=1.12.0",
    "azure-keyvault-secrets>=4.7.0",
    "openai>=1.0.0",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio"]
```

#### Installing Plugin Dependencies

**Option 1: Use `--install-plugin-deps` flag (recommended for development)**

```bash
hexdag-studio ./pipelines/ --plugin ./my_plugin --install-plugin-deps
```

This automatically detects and uses the appropriate package manager:
- Uses `uv pip install` if uv is available and working
- Falls back to `pip install` otherwise

Plugins are installed in editable mode (`-e`) so changes reflect immediately.

**Option 2: Manual installation**

```bash
# With uv
uv pip install -e ./my_plugin

# With pip
pip install -e ./my_plugin
```

**Option 3: Add to project dependencies**

```toml
# your_project/pyproject.toml
[project]
dependencies = [
    "hexdag>=0.5.0",
    "hexdag-studio>=0.1.0",
]

[project.optional-dependencies]
azure = ["hexdag-plugin-azure>=0.2.0"]

# Or for local plugins:
[tool.uv.sources]
hexdag-plugin-azure = { path = "../hexdag_plugins/azure", editable = true }
```

Then install with:
```bash
uv sync --extra azure
```

### Missing Dependencies

If a plugin has missing dependencies, Studio shows a helpful error:

```
Plugin 'azure' requires missing dependency: openai
  Install with: pip install openai
```

Use `--install-plugin-deps` to auto-install, or install manually.

## Building a Plugin (Step-by-Step Guide)

This guide shows how to create a hexDAG plugin **outside** the main `hexdag_plugins` directory.

### 1. Create Plugin Directory

```bash
mkdir -p ~/my-hexdag-plugins/my_custom_plugin
cd ~/my-hexdag-plugins/my_custom_plugin
```

### 2. Create `pyproject.toml`

```toml
[project]
name = "hexdag-plugin-my-custom"
version = "0.1.0"
description = "My custom hexDAG plugin"
requires-python = ">=3.12"
dependencies = [
    "hexdag>=0.5.0",
    # Add your plugin's dependencies here
    # "httpx>=0.24.0",
    # "pandas>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### 3. Create `__init__.py`

```python
"""My custom hexDAG plugin.

Provides custom adapters and nodes for specialized workflows.
"""

from my_custom_plugin.adapters import MyDatabaseAdapter, MyAPIAdapter
from my_custom_plugin.nodes import MyProcessorNode

__version__ = "0.1.0"
__all__ = [
    "MyDatabaseAdapter",
    "MyAPIAdapter",
    "MyProcessorNode",
]
```

### 4. Create Adapters (`adapters.py`)

```python
"""Custom adapters for my plugin."""

import os
from typing import Any
from hexdag.kernel.ports.database import Database

class MyDatabaseAdapter(Database):
    """Custom database adapter implementing Database port."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "mydb",
        username: str = "user",
        password: str | None = None,
        **kwargs: Any,
    ) -> None:
        password = password or os.environ.get("MY_DB_PASSWORD", "")
        self.connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"

    async def aexecute_query(self, query: str) -> list[dict]:
        # Implementation
        ...


class MyAPIAdapter:
    """Custom API adapter."""

    def __init__(
        self,
        endpoint: str,
        api_key: str | None = None,
        timeout: float = 30.0,
        **kwargs: Any,
    ) -> None:
        self.endpoint = endpoint
        self.api_key = api_key or os.environ.get("MY_API_KEY", "")
        self.timeout = timeout

    async def aresponse(self, messages: list) -> str:
        # Implementation
        ...
```

Use in YAML: `adapter: my_plugin.adapters.MyDatabaseAdapter`

### 5. Create Nodes (`nodes.py`)

```python
"""Custom nodes for my plugin."""

from hexdag.stdlib.nodes import BaseNodeFactory

class MyProcessorNode(BaseNodeFactory):
    """Custom data processor node."""

    def __call__(
        self,
        name: str,
        batch_size: int = 100,
        transform_type: str = "default",
        **kwargs,
    ):
        # Node factory implementation
        ...
```

Use in YAML: `kind: my_plugin.nodes.MyProcessorNode`

### 6. Final Directory Structure

```
~/my-hexdag-plugins/my_custom_plugin/
├── __init__.py          # Exports via __all__
├── pyproject.toml       # Dependencies
├── adapters.py          # Adapter classes implementing port interfaces
├── nodes.py             # Node classes extending BaseNodeFactory
└── tests/               # Optional tests
    └── test_adapters.py
```

### 7. Install and Use

```bash
# Install the plugin (editable mode for development)
uv pip install -e ~/my-hexdag-plugins/my_custom_plugin

# Or use hexdag-studio's auto-install
hexdag-studio ./pipelines/ --plugin ~/my-hexdag-plugins/my_custom_plugin --install-plugin-deps

# Load the plugin in Studio
hexdag-studio ./pipelines/ --plugin ~/my-hexdag-plugins/my_custom_plugin
```

### 8. Use in YAML Pipelines

```yaml
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: my-workflow
spec:
  ports:
    database:
      adapter: my_database
      config:
        host: db.example.com
        database: production
    llm:
      adapter: my_api
      config:
        endpoint: https://api.example.com/v1
  nodes:
    - kind: my_plugin:my_processor
      metadata:
        name: process_data
      spec:
        batch_size: 500
        transform_type: advanced
```

---

## Plugin Development Reference

### Creating Adapters

**IMPORTANT:** Use explicit typed parameters in `__init__` for config schema generation.

```python
# my_plugin/__init__.py
import os
from typing import Any
from hexdag.kernel.ports.llm import LLM

class MyLLMAdapter(LLM):
    """Custom LLM adapter implementing the LLM port interface."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "default-model",
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,  # Keep for forward compatibility
    ) -> None:
        """Initialize adapter.

        Args:
            api_key: API key for authentication.
            model: Model identifier to use.
            temperature: Sampling temperature (0.0-2.0).
            max_tokens: Maximum tokens in response.
            **kwargs: Additional options.
        """
        self.api_key = api_key or os.environ.get("MY_LLM_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def aresponse(self, messages):
        # Implementation here
        ...

__all__ = ["MyLLMAdapter"]
```

Use in YAML: `adapter: my_plugin.MyLLMAdapter`

### Why Explicit Parameters Matter

The Studio UI generates configuration forms from your adapter's `__init__` signature using `SchemaGenerator`.

**✅ Explicit parameters → Rich config UI:**
```python
def __init__(self, model: str = "gpt-4", temperature: float = 0.7):
    ...
# Studio shows: model (text input), temperature (number input)
```

**❌ kwargs-only → Empty config UI:**
```python
def __init__(self, **kwargs):
    self.model = kwargs.get("model", "gpt-4")
# Studio shows: nothing! Users can't configure your adapter.
```

### Best Practices

1. **Always use explicit typed parameters** for all configurable options
2. **Keep `**kwargs` at the end** for forward compatibility
3. **Add docstrings** - descriptions are extracted for UI tooltips
4. **Use `| None`** for optional parameters with `None` default
5. **Use `Literal` types** for enum-like options (generates dropdowns)

```python
from typing import Literal

def __init__(
    self,
    mode: Literal["fast", "balanced", "quality"] = "balanced",
    timeout: float | None = None,
    **kwargs: Any,
) -> None:
    """
    Args:
        mode: Processing mode (fast/balanced/quality).
        timeout: Request timeout in seconds. None for no timeout.
    """
```

### Creating Nodes

```python
from hexdag.stdlib.nodes import BaseNodeFactory

class MyNode(BaseNodeFactory):
    """Custom processing node."""

    def __call__(
        self,
        name: str,
        config_option: str = "default",
        threshold: float = 0.5,
        **kwargs,
    ):
        # Node factory implementation
        ...

__all__ = ["MyNode"]
```

## License

MIT
