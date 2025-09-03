# HexDAG Component Registry

A lightweight, decorator-based component registry for the hexDAG framework, built on [pluggy](https://pluggy.readthedocs.io/).

## Overview

The registry provides a unified way to register and discover components (nodes, adapters, tools, etc.) for both core hexDAG and third-party plugins. It uses decorators for automatic registration and pluggy for plugin discovery.

## Quick Start

### Registering Core Components

```python
from hexai.core.registry import node

@node()  # Automatically registers in 'core' namespace
class PassthroughNode:
    """A simple node that passes data through."""
    def execute(self, data):
        return data
```

### Creating Plugin Components

```python
from hexai.core.registry import node, tool

@node(namespace='my_plugin')
class AnalyzerNode:
    """Custom analyzer node."""
    def execute(self, data):
        return analyze(data)

@tool(namespace='my_plugin')
class DataFetcher:
    """Fetches data from external sources."""
    def fetch(self, url):
        return fetch_data(url)
```

### Accessing Components

```python
from hexai.core.registry import registry

# Get core component (searches 'core' namespace first)
passthrough = registry.get('passthrough_node')

# Get plugin component
analyzer = registry.get('analyzer_node', namespace='my_plugin')

# List all components
components = registry.list_components()
```

## Features

### 🎯 Unified Decorator API
Same decorators for core and plugin components - just specify the namespace.

### 🔒 Automatic Core Protection
Core components cannot be overridden. Attempts to shadow them trigger warnings:
```
⚠️ Component 'passthrough_node' shadows CORE component!
   Core version remains at 'core:passthrough_node'
```

### 📦 Clean Namespace Isolation
- `core`: Protected hexDAG components
- `your_plugin`: Your plugin's components
- No accidental collisions between plugins

### 🚀 Powered by Pluggy
- Battle-tested in pytest ecosystem
- Automatic discovery via setuptools entry points
- Hook-based plugin system

### 💡 Smart Type Inference
Component type inferred from class name:
- `*Node` → NODE
- `*Adapter` → ADAPTER
- `*Tool` → TOOL
- `*Policy` → POLICY
- `*Memory` → MEMORY
- `*Observer` → OBSERVER

## Component Types

### Decorators Available

```python
from hexai.core.registry import (
    component,  # Generic decorator
    node,       # For processing nodes
    adapter,    # For data adapters
    tool,       # For utility tools
    policy,     # For execution policies
    memory,     # For memory stores
    observer,   # For event observers
)
```

### Example Usage

```python
@node()
class DataProcessor:
    """Processes incoming data."""
    pass

@adapter()
class PostgresAdapter:
    """Connects to PostgreSQL."""
    pass

@tool()
class WebScraper:
    """Scrapes web content."""
    pass
```

## Plugin Development

### 1. Create Your Plugin

```python
# my_plugin/nodes.py
from hexai.core.registry import node

@node(namespace='my_plugin')
class CustomNode:
    def execute(self, data):
        return process(data)
```

### 2. Create Entry Point

```python
# my_plugin/__init__.py
def register():
    """Entry point for plugin registration."""
    from . import nodes  # Import triggers decorators
```

### 3. Configure in pyproject.toml

```toml
[project.entry-points."hexdag.plugins"]
my_plugin = "my_plugin:register"
```

### 4. Install and Use

```bash
pip install my-plugin
```

```python
from hexai.core.registry import registry

# Your plugin is automatically discovered!
node = registry.get('custom_node', namespace='my_plugin')
```

## API Reference

### Registry Methods

#### `get(name, namespace=None, component_type=None)`
Get a component by name. Searches core namespace first if namespace not specified.

#### `list_components(namespace=None, component_type=None)`
List all registered components, optionally filtered.

#### `get_metadata(name, namespace='core')`
Get metadata for a component.

#### `list_namespaces()`
List all registered namespaces.

### Decorator Parameters

All decorators accept these parameters:

- `name`: Component name (default: snake_case of class name)
- `namespace`: Component namespace (default: 'core')
- `description`: Component description (default: class docstring)
- `tags`: Set of tags for categorization
- `author`: Component author
- `dependencies`: Set of component dependencies
- `replaceable`: Whether component can be replaced
- `version`: Component version

## Architecture

```
Component Registry (Singleton)
├── Core Components (Protected)
│   ├── PassthroughNode
│   └── LoggingNode
└── Plugin Components (Namespaced)
    ├── my_plugin:CustomNode
    └── other_plugin:AnalyzerNode
```

### Key Design Decisions

1. **Decorator-based**: Consistent, Pythonic API
2. **Namespace isolation**: Prevents conflicts
3. **Core protection**: Ensures system stability
4. **Lazy instantiation**: Components created on-demand
5. **Pluggy integration**: Leverages proven plugin system

## Comparison with Previous Implementation

| Aspect | Old (1600+ lines) | New (450 lines) |
|--------|------------------|-----------------|
| Complexity | High (frame inspection, complex locks) | Low (simple decorators) |
| Security | Frame inspection vulnerabilities | Namespace-based protection |
| Plugin System | Custom implementation | Battle-tested pluggy |
| API | Mixed (manual + decorators) | Unified decorators |
| Code Size | 1,670 lines | ~450 lines (-73%) |

## Best Practices

### ✅ DO

- Use descriptive component names
- Specify namespace for plugins
- Add docstrings (become descriptions)
- Use type-specific decorators (`@node`, `@tool`, etc.)
- Handle errors gracefully in components

### ❌ DON'T

- Try to override core components
- Use 'core' namespace for plugins
- Forget to specify namespace in plugins
- Create circular dependencies
- Instantiate components manually

## Migration from Old Registry

### Before
```python
registry.register(
    name='my_node',
    component=MyNode(),
    component_type=ComponentType.NODE,
    namespace='my_plugin',
    metadata=ComponentMetadata(...)
)
```

### After
```python
@node(namespace='my_plugin')
class MyNode:
    pass
```

## Troubleshooting

### Component Not Found
- Check namespace is correct
- Ensure decorator was applied
- Verify plugin is installed

### Shadow Warning
- This is intentional - core components are protected
- Use explicit namespace to access plugin version

### Import Errors
- Ensure pluggy is installed: `pip install pluggy`
- Check circular imports in plugin modules

## Contributing

When adding new core components:

1. Add to appropriate module in `hexai/core/`
2. Use decorator without namespace (defaults to 'core')
3. Set `replaceable=False` for critical components
4. Add tests

## License

Part of the hexDAG project. See main project LICENSE.
