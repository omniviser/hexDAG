# HexDAG Component Registry

A simplified, string-based component registry for the HexDAG framework. Clean API, no bloat, easy to use.

## Quick Start

```python
from hexai.core.registry import node, tool, registry

# Register a class component (instantiated on get)
@node(namespace="user")
class DataProcessor:
    def __init__(self, batch_size=32):
        self.batch_size = batch_size

    def execute(self, data):
        return process_in_batches(data, self.batch_size)

# Register a function component (returned as-is, not called)
@tool(namespace="user")
def fetch_data(url: str):
    return requests.get(url).json()

# Get components from registry
processor = registry.get("data_processor", namespace="user")  # Returns new instance
print(processor.batch_size)  # 32

processor = registry.get("data_processor", namespace="user", batch_size=64)  # Custom args
print(processor.batch_size)  # 64

fetcher = registry.get("fetch_data", namespace="user")  # Returns function itself
data = fetcher("https://api.example.com")  # Call it with your args
```

## Key Concepts

### Everything is Strings
No need to import enums or types. All decorators accept strings:

```python
@node(namespace="my_plugin")  # String namespace
class MyNode:
    pass

@component("tool", namespace="utilities")  # String component type
def my_tool():
    pass
```

### Class vs Function Components

**Classes** are instantiated on each `get()`:
```python
@node(namespace="user")
class StatefulProcessor:
    def __init__(self, config=None):
        self.state = {}
        self.config = config or {}

# Each get() creates a new instance
proc1 = registry.get("stateful_processor", namespace="user")
proc2 = registry.get("stateful_processor", namespace="user")
assert proc1 is not proc2  # Different instances
```

**Functions** are returned as-is (not called):
```python
@tool(namespace="user")
def transform_data(data, scale=1.0):
    return [x * scale for x in data]

# get() returns the function itself
transform = registry.get("transform_data", namespace="user")
result = transform([1, 2, 3], scale=2.0)  # Call it yourself
```

## Namespaces

### System Namespaces
- `"user"` - **Default namespace** for user components
- `"core"` - Protected namespace for framework components (requires privilege)
- `"plugin"` - Standard namespace for plugin components

### Custom Namespaces
Any other string is a custom namespace:

```python
@node(namespace="analytics")
class StatsNode:
    pass

@tool(namespace="cloud_services")
def upload_to_s3(file):
    pass
```

## Component Types

All specified as strings:
- `"node"` - Processing nodes
- `"tool"` - Utility functions
- `"adapter"` - Data adapters
- `"policy"` - Execution policies
- `"memory"` - Storage components
- `"observer"` - Event observers

## Decorators

### Basic Decorators
```python
from hexai.core.registry import node, tool, adapter, policy, memory, observer

@node(namespace="user")  # Default namespace is "user"
class MyNode:
    pass

@tool(name="custom_name", namespace="utilities")
def my_tool():
    pass
```

### Node Subtypes
```python
from hexai.core.registry import function_node, llm_node, agent_node

@function_node(namespace="user")
class DataTransformer:
    pass

@llm_node(namespace="ai")
class ChatNode:
    pass
```

### Generic Decorator
```python
from hexai.core.registry import component

@component("node", namespace="user", subtype="custom")
class CustomNode:
    pass
```

## Registry API

### Getting Components
```python
from hexai.core.registry import registry

# Basic get
node = registry.get("my_node", namespace="user")

# With namespace in name
node = registry.get("user:my_node")

# With kwargs (for class instantiation)
node = registry.get("my_node", namespace="user", config={'debug': True})
```

### Listing Components
```python
# List all
all_components = registry.list_components()

# Filter by type
nodes = registry.list_components(component_type="node")

# Filter by namespace
user_components = registry.list_components(namespace="user")
```

### Component Metadata
```python
metadata = registry.get_metadata("my_node", namespace="user")
print(f"Name: {metadata.name}")
print(f"Type: {metadata.component_type}")
print(f"Description: {metadata.description}")
```

## Plugin Development

Create components in custom namespaces:

```python
# my_plugin/nodes.py
from hexai.core.registry import node, tool

NAMESPACE = "my_plugin"

@node(namespace=NAMESPACE)
class AnalysisNode:
    """Performs custom analysis."""
    def execute(self, data):
        return analyze(data)

@tool(namespace=NAMESPACE)
def preprocess(data):
    """Preprocessing utility."""
    return clean(data)
```

## Implementation Details

### Architecture
- **~260 lines of code** (down from 500+)
- **5 files** (down from 9)
- Thread-safe with RLock
- No external dependencies beyond Python stdlib

### Files
- `registry.py` - Core registry logic (125 lines)
- `decorators.py` - Decorator functions (36 lines)
- `metadata.py` - Component metadata (27 lines)
- `plugin_loader.py` - Plugin discovery (62 lines)
- `types.py` - Type definitions (21 lines)

### How It Works
1. Decorators register components immediately when imported
2. Registry stores metadata, not instances
3. `get()` creates instances (classes) or returns as-is (functions)
4. Namespaces provide isolation
5. System namespaces get special handling

## Best Practices

1. **Use "user" namespace by default** - It's the default for a reason
2. **Document your components** - Docstrings become descriptions
3. **Functions for stateless operations** - Simpler and more efficient
4. **Classes for stateful components** - When you need state or initialization
5. **Custom namespaces for plugins** - Avoid conflicts
6. **Never use "core" namespace** - Unless you're developing HexDAG itself

## Common Pitfalls

### Function Called Unexpectedly
❌ **Wrong**: Expecting function to be called automatically
```python
@tool(namespace="user")
def fetch_data(url):
    return data

# This returns the function, not the data!
data = registry.get("fetch_data", namespace="user")
```

✅ **Right**: Call the function yourself
```python
fetcher = registry.get("fetch_data", namespace="user")
data = fetcher("https://api.example.com")
```

### Class Not Instantiated
❌ **Wrong**: Using function decorator for a class
```python
@tool(namespace="user")  # tool decorator is for functions!
class MyTool:
    pass
```

✅ **Right**: Use appropriate decorator
```python
@node(namespace="user")  # node decorator for classes
class MyNode:
    pass
```

### Namespace Conflicts
❌ **Wrong**: Using "core" namespace
```python
@node(namespace="core")  # Will fail without privilege
class MyNode:
    pass
```

✅ **Right**: Use "user" or custom namespace
```python
@node(namespace="user")  # Or namespace="my_plugin"
class MyNode:
    pass
```

## Testing

The registry has comprehensive test coverage:

```bash
# Run tests
pytest tests/hexai/core/registry/

# Coverage
- decorators.py: 100% coverage
- registry.py: 88% coverage
- metadata.py: 96% coverage
- Total: 60+ passing tests
```

## Summary

The new registry delivers:
- ✅ **Simple string-based API** - No enums to import
- ✅ **Clear semantics** - Classes instantiated, functions returned
- ✅ **Minimal code** - 260 lines total
- ✅ **Thread-safe** - RLock protection
- ✅ **Well-tested** - 60+ tests, high coverage
- ✅ **Plugin-friendly** - Custom namespaces
- ✅ **No surprises** - Predictable behavior
