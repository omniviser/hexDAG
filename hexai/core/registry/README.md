# Component Registry

Bootstrap-based component registry with declarative YAML manifests and namespace isolation.

## Overview

The registry follows a **bootstrap-based architecture** similar to Django's app registry:
1. **Decorators add metadata** - No auto-registration or import side effects
2. **Manifest declares components** - YAML file lists what modules to load
3. **Bootstrap initializes** - Explicit initialization populates the registry
4. **Immutable after bootstrap** - Registry is read-only in production

## Quick Start

### 1. Define Components

```python
# my_components.py
from hexai.core.registry.decorators import node, tool

@node(name="data_processor", description="Processes data")
class DataProcessor:
    """Component with metadata - NOT auto-registered."""
    def execute(self, data):
        return process(data)

@tool(name="utility_function")
def utility_function(x):
    """Tool with metadata - NOT auto-registered."""
    return x * 2
```

### 2. Create Manifest

```yaml
# hexdag_manifest.yaml or hexai/core/component_manifest.yaml
components:
  - namespace: core
    module: hexai.core.nodes
  - namespace: user
    module: my_components

config:
  dev_mode: false
  search_priority:
    - core
    - user
```

### 3. Bootstrap and Use

```python
from hexai.core import init_hexdag, registry

# Initialize registry from manifest
init_hexdag()  # or bootstrap_registry(manifest_path="custom.yaml")

# Now use components
processor = registry.get("data_processor")
result = processor.execute(data)
```

## Bootstrap API

### Initialization Functions

#### `init_hexdag(dev_mode=False)`
Simple one-liner initialization with defaults.

```python
from hexai.core import init_hexdag
init_hexdag(dev_mode=True)  # For development
```

#### `bootstrap_registry(manifest_path=None, dev_mode=None)`
Full control over bootstrap process.

```python
from hexai.core.bootstrap import bootstrap_registry
bootstrap_registry(
    manifest_path="config/production.yaml",
    dev_mode=False
)
```

#### `registry.bootstrap(manifest, dev_mode=False)`
Low-level bootstrap on registry instance.

```python
from hexai.core.registry import registry
from hexai.core.registry.manifest import ComponentManifest

manifest = ComponentManifest([
    {"namespace": "core", "module": "hexai.core.nodes"}
])
registry.bootstrap(manifest)
```

## Registry Operations

### Core Methods

#### `get(name, namespace=None, **kwargs)`
Get and instantiate a component.

```python
# With namespace search priority
tool = registry.get("my_tool")

# With explicit namespace
tool = registry.get("my_tool", namespace="user")

# With qualified name
tool = registry.get("user:my_tool")

# With constructor arguments
tool = registry.get("my_tool", config={"debug": True})
```

#### `get_metadata(name, namespace=None)`
Get component metadata without instantiation.

```python
metadata = registry.get_metadata("my_tool")
print(f"Type: {metadata.component_type}")
print(f"Description: {metadata.description}")
```

#### `list_components(component_type=None, namespace=None)`
List registered components.

```python
# All components
all_components = registry.list_components()

# Filter by type
tools = registry.list_components(component_type="tool")

# Filter by namespace
user_components = registry.list_components(namespace="user")
```

### Registry State

#### `ready`
Check if registry has been bootstrapped.

```python
if not registry.ready:
    init_hexdag()
```

#### `manifest`
Access the current manifest.

```python
if registry.manifest:
    print(f"Loaded from: {registry.manifest.entries}")
```

#### `reset(namespace=None)`
Reset registry (mainly for testing).

```python
# Reset everything
registry.reset()

# Reset specific namespace
registry.reset(namespace="test")
```

## Manifest Structure

### YAML Format

```yaml
# Component declarations
components:
  - namespace: core
    module: hexai.core.nodes
  - namespace: plugin
    module: my_plugin.components
  - namespace: user
    module: myproject.components

# Optional configuration
config:
  dev_mode: false  # Allow post-bootstrap registration
  search_priority:  # Namespace resolution order
    - core
    - user
    - plugin
  validation:
    prevent_shadowing: false
    require_descriptions: false
```

### Loading Manifests

```python
from hexai.core.registry.manifest import load_manifest_from_yaml

# From YAML file
manifest = load_manifest_from_yaml("my_manifest.yaml")

# From Python list
from hexai.core.registry.manifest import ComponentManifest
manifest = ComponentManifest([
    {"namespace": "core", "module": "hexai.core.nodes"}
])
```

## Component Discovery

### Default Discovery
Modules are scanned for decorated components automatically.

```python
# In your module
@node(name="my_node")
class MyNode:
    pass

# Automatically discovered during bootstrap
```

### Custom Registration Hook
Modules can define custom registration logic.

```python
# my_module.py
def register_components(registry, namespace):
    """Custom registration logic."""
    # Conditional registration
    if os.getenv("ENABLE_EXPERIMENTAL"):
        registry.register(
            name="experimental",
            component=ExperimentalNode,
            namespace=namespace,
            privileged=(namespace == "core")
        )
```

## Development Mode

### Enable Dev Mode

```python
# Via init
init_hexdag(dev_mode=True)

# Via environment
export HEXDAG_DEV_MODE=true

# Via manifest
config:
  dev_mode: true
```

### Dev Mode Features
- Post-bootstrap registration allowed
- Helpful for interactive development
- Testing with mock components

## Namespaces

- `core` - Protected system components (requires privilege)
- `user` - Default for user components
- `plugin` - Third-party plugins
- `test` - Testing components
- Custom - Any alphanumeric string

## Thread Safety

All operations are thread-safe:
- Multiple concurrent reads allowed
- Writes have exclusive access
- Bootstrap is single-threaded

## Architecture

```
manifest.py       - YAML manifest loading
discovery.py      - Component discovery system
bootstrap.py      - Bootstrap utilities
registry.py       - Core registry (immutable after bootstrap)
decorators.py     - Metadata-only decorators
models.py         - Data models
locks.py          - Thread safety
exceptions.py     - Custom exceptions
```

## Plugin Dependency Handling

The registry handles plugin dependencies through **graceful degradation**, ensuring the core framework remains functional even when optional plugins have missing dependencies.

### How It Works

#### Module Classification

Modules are classified into two categories during bootstrap:

**Core Modules** (`hexai.core.*`, `hexai.tools.*`):
- **MUST** load successfully
- Import failure → Bootstrap **FAILS** with exception
- Essential framework components

**Plugin Modules** (everything else):
- **OPTIONAL** - can fail gracefully
- Import failure → Logged as WARNING and **SKIPPED**
- Bootstrap continues successfully

#### Loading Process

```
Config → Bootstrap → Import Module → Register Components
                         ↓ (if fails)
                   Log Warning & Skip (plugins only)
```

### Example: Plugin with Missing Dependencies

```toml
# hexdag.toml
plugins = [
    "hexai.adapters.mock",              # ✅ No external dependencies
    "hexai_plugins.mysql_adapter",      # ⚠️  Requires pymysql
    "hexai.adapters.llm.openai_adapter" # ⚠️  Requires openai
]
```

**If pymysql is not installed:**
```
INFO: Registered 4 components from hexai.adapters.mock into namespace 'plugin'
WARNING: Optional module hexai_plugins.mysql_adapter not available: No module named 'pymysql'
INFO: Registered 1 component from hexai.adapters.llm.openai_adapter into namespace 'plugin'
INFO: Bootstrap complete: 12 components registered
```

The framework continues working normally with the available plugins.

### Installing Plugin Dependencies

#### Option 1: Install Plugin Package
```bash
# Plugin package includes dependencies in pyproject.toml
pip install hexdag-mysql-adapter  # Automatically installs pymysql
```

#### Option 2: Install with Extras
```bash
# If defined in hexdag's pyproject.toml
pip install hexdag[mysql,openai]
```

#### Option 3: Manual Installation
```bash
pip install pymysql openai
# Then plugins will be available
```

### For Plugin Developers

#### 1. Declare Dependencies in pyproject.toml

```toml
[project]
name = "hexdag-mysql-adapter"
dependencies = [
    "pymysql>=1.1.0",
    "cryptography>=41.0.0"
]

[tool.hexdag.plugin]
name = "mysql"
module = "hexai_plugins.mysql_adapter"
port = "database"
description = "Production-ready MySQL adapter"
requires_env = ["MYSQL_HOST", "MYSQL_USER", "MYSQL_PASSWORD"]
```

#### 2. Handle Runtime Requirements Gracefully

```python
from hexai.core.registry import adapter

@adapter(
    name="mysql_database",
    namespace="plugin",
    port="database"
)
class MySQLAdapter:
    """MySQL database adapter."""

    def __init__(self, host: str, user: str, password: str):
        """Initialize adapter.

        Raises:
            ValueError: If required credentials are missing
        """
        if not all([host, user, password]):
            raise ValueError(
                "MySQL adapter requires host, user, and password. "
                "Set MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD environment variables."
            )
        self.connection = pymysql.connect(...)
```

#### 3. Provide Clear Documentation

```markdown
# hexdag-mysql-adapter

## Installation

```bash
pip install hexdag-mysql-adapter
```

## Configuration

```bash
export MYSQL_HOST=localhost
export MYSQL_USER=root
export MYSQL_PASSWORD=secret
export MYSQL_DATABASE=mydb
```

## Usage

```toml
# hexdag.toml
plugins = ["hexai_plugins.mysql_adapter"]
```
```

### Dependency Checking Behavior

The registry's `_check_plugin_requirements()` method:

1. **Checks if module exists** using `importlib.util.find_spec()`
2. **Does NOT** install missing dependencies automatically
3. **Does NOT** parse or validate `pyproject.toml` dependencies
4. **Does NOT** check Python version compatibility
5. **Allows** import to fail and handles the exception gracefully

This is **by design** following Python packaging best practices:
- Users explicitly control what gets installed
- No automatic package installation (security concern)
- Dependencies declared in package metadata
- Framework degrades gracefully when components missing

### Best Practices

#### For Users

1. **Check logs** for skipped plugins during bootstrap
2. **Install required plugins** before running production code
3. **Use virtual environments** to isolate dependencies
4. **Document plugin requirements** in your project README

#### For Plugin Developers

1. **Declare all dependencies** in `pyproject.toml`
2. **Document environment variables** and configuration
3. **Handle missing config gracefully** with helpful error messages
4. **Test plugin isolation** - ensure it doesn't break core framework
5. **Use namespace="plugin"** for third-party plugins

### Limitations

Current implementation does **NOT**:
- Automatically install missing dependencies
- Parse `pyproject.toml` to check dependencies before import
- Validate plugin metadata before loading
- Resolve dependency conflicts
- Check Python version compatibility

These are intentional design decisions prioritizing:
- User control over installations
- Security (no automatic pip installs)
- Simplicity
- Standard Python packaging practices

## Testing

```bash
pytest tests/hexai/core/registry/
```

The bootstrap architecture makes testing much cleaner:
- Each test can have its own manifest
- Clean reset between tests
- No import order dependencies
- Easy to mock components
