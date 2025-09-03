# HexDAG Component Registry

A unified, decorator-based component registry for the hexDAG framework, built on [pluggy](https://pluggy.readthedocs.io/).

## 🎯 Overview

The HexDAG registry provides a unified way to register and discover components (nodes, adapters, tools, etc.) for both core hexDAG and third-party plugins, using decorators for automatic registration and pluggy for plugin discovery.

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

## ✨ Key Features

### 1. **Automatic Core Registration**
Core components are automatically registered at startup without any manual registration code. No need to explicitly register core components - they're available immediately.

### 2. **Unified Decorator API**
Same decorators for core and plugin components - just specify the namespace. Simple, intuitive API that Python developers already understand.

### 3. **Namespace Isolation**
- `core`: Protected hexDAG components
- `your_plugin`: Your plugin's components
- No accidental collisions between plugins
- Components with same names can coexist in different namespaces

### 4. **Core Component Protection**
Core components cannot be overridden. Attempts to shadow them trigger warnings:
```
⚠️ Component 'passthrough_node' shadows CORE component!
   Core version remains at 'core:passthrough_node'
   Plugin version will be at 'my_plugin:passthrough_node'
```
Both versions remain accessible via their full names.

### 5. **Lazy Instantiation**
Components are only instantiated when first accessed, improving performance:
- Reduces startup time
- Saves memory for unused components
- Singleton pattern ensures single instance per component

### 6. **Rich Metadata & Dependencies**
Full support for versioning, authorship, tags, and dependency tracking:
```python
@node(
    namespace="analytics",
    version="2.5.0",
    author="Data Team",
    tags={"ml", "prediction"},
    dependencies={"core:pass_through_node"}
)
```

### 7. **Plugin Discovery via Entry Points**
- Battle-tested pluggy framework (used by pytest)
- Automatic discovery via setuptools entry points
- No manual plugin loading required

### 8. **Smart Type Inference**
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

## 📊 Comparison with Previous Implementation

| Aspect | Old System (1600+ lines) | New System (~450 lines) | Improvement |
|--------|--------------------------|-------------------------|-------------|
| **Code Size** | 1,670 lines | ~450 lines | **73% reduction** |
| **Complexity** | High (graphs, double locking, frame inspection) | Low (simple registry, decorators) | Much simpler |
| **Security** | Frame inspection vulnerabilities | No frame inspection | **More secure** |
| **Plugin System** | Custom implementation | Battle-tested pluggy | **Production ready** |
| **API** | Mixed (manual + decorators) | Unified decorators | **More intuitive** |
| **Registration** | Manual, complex | Automatic via decorators | **Developer friendly** |
| **Type Safety** | Partial | Full mypy support | **Better IDE support** |
| **Performance** | Eager loading | Lazy instantiation | **Faster startup** |
| **Testing** | Limited | 92+ tests | **Better coverage** |

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

## 🧪 Testing

The registry includes comprehensive test coverage:

```bash
# Run all registry tests
pytest tests/hexai/core/registry/

# Test coverage:
# - test_registry.py: 21 tests
# - test_decorators.py: 33 tests
# - test_metadata.py: 12 tests
# - test_types.py: 14 tests
# - test_discovery.py: 12 tests
# Total: 92+ tests with full coverage
```

## 📝 Examples

Complete examples are available in the `examples/` directory:

- **`example_21_plugin_system.py`** - Comprehensive demonstration of all features
- **`test_plugin_system.py`** - Test suite verifying all properties (6/6 tests passing)
- **`example_plugin_package/`** - Complete plugin package with modern `pyproject.toml`
- **`create_plugin_template.py`** - Tool to generate new plugins from template

## 🎉 Summary

The new registry system successfully delivers:

- ✅ **Automatic core registration** at startup
- ✅ **Easy plugin development** with decorators
- ✅ **Warnings when overriding** core components
- ✅ **Namespace isolation** for safety
- ✅ **Lazy instantiation** for performance
- ✅ **Rich metadata** and dependency tracking
- ✅ **Based on proven technology** (pluggy, used by pytest)
- ✅ **73% code reduction** (450 lines vs 1600+)
- ✅ **Full type safety** with mypy support
- ✅ **Production ready** with comprehensive testing

This is a production-ready component registry that provides a solid foundation for extending HexDAG with custom components while maintaining simplicity, safety, and performance.

## License

Part of the hexDAG project. See main project LICENSE.
