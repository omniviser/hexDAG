# Component Registry

Bootstrap-based component registry with declarative YAML manifests and namespace isolation.

## Architecture (Simplified 2025)

The registry has been **simplified from 2,668 → 1,906 lines (28.5% reduction)** by undoing over-engineered abstractions:

**What was removed:**
- ❌ `ComponentStore` (414 lines) - Storage now inline in ComponentRegistry
- ❌ `BootstrapManager` (302 lines) - Lifecycle management now inline
- ❌ `ReadWriteLock` (144 lines) - Replaced with simple `threading.Lock`
- ❌ `validation.py` (212 lines) - Merged into `models.py`

**Current structure (7 files):**
- ✅ `registry.py` (538 lines) - All-in-one registry class
- ✅ `models.py` (387 lines) - Types + validation logic
- ✅ `decorators.py` (392 lines) - Component decorators
- ✅ `discovery.py` (245 lines) - Auto-discovery with two-phase registration
- ✅ `introspection.py` (185 lines) - **Optional** utilities for testing/CLI
- ✅ `exceptions.py` (82 lines) - Error types
- ✅ `__init__.py` (77 lines) - Public API

**Key principle:** A single cohesive 500-line registry class is NOT a "god class" - it's appropriate for a registry. The previous "separation of concerns" refactoring made it worse, not better.

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
from hexdag.core.registry.decorators import node, tool

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
# hexdag_manifest.yaml or hexdag/core/component_manifest.yaml
components:
  - namespace: core
    module: hexdag.core.nodes
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
from hexdag.core import init_hexdag, registry

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
from hexdag.core import init_hexdag
init_hexdag(dev_mode=True)  # For development
```

#### `bootstrap_registry(manifest_path=None, dev_mode=None)`
Full control over bootstrap process.

```python
from hexdag.core.bootstrap import bootstrap_registry
bootstrap_registry(
    manifest_path="config/production.yaml",
    dev_mode=False
)
```

#### `registry.bootstrap(manifest, dev_mode=False)`
Low-level bootstrap on registry instance.

```python
from hexdag.core.registry import registry
from hexdag.core.registry.manifest import ComponentManifest

manifest = ComponentManifest([
    {"namespace": "core", "module": "hexdag.core.nodes"}
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
    module: hexdag.core.nodes
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
from hexdag.core.registry.manifest import load_manifest_from_yaml

# From YAML file
manifest = load_manifest_from_yaml("my_manifest.yaml")

# From Python list
from hexdag.core.registry.manifest import ComponentManifest
manifest = ComponentManifest([
    {"namespace": "core", "module": "hexdag.core.nodes"}
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

Bootstrap uses a simple `threading.Lock` for thread safety during initialization.
After bootstrap, the registry is immutable and thread-safe by design (no writes).

## File Structure

```
registry.py       - All-in-one registry (storage + lifecycle + bootstrap)
models.py         - Type definitions + validation logic
decorators.py     - Metadata-only decorators (@node, @adapter, etc)
discovery.py      - Component auto-discovery with two-phase registration
introspection.py  - Optional utilities for testing/CLI (not core)
exceptions.py     - Custom exception types
__init__.py       - Public API exports
```

## Testing

```bash
pytest tests/hexdag/core/registry/
```

The bootstrap architecture makes testing much cleaner:
- Each test can have its own manifest
- Clean reset between tests
- No import order dependencies
- Easy to mock components
