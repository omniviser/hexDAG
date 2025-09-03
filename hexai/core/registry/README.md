# Registry System

## Overview

The hexDAG Registry System provides a centralized, append-only, singleton-based component discovery and management framework. Built on Pydantic validation and following hexagonal architecture principles, it enables dynamic component registration with full plugin support through isolated namespaces. The registry follows an append-only pattern for production stability, with components marked as replaceable when needed.

## Key Features

- **Singleton Registry**: One central registry for all components
- **Namespace Isolation**: Plugins get their own namespaces to prevent conflicts
- **Type-Safe Metadata**: Full Pydantic validation for component metadata
- **Plugin Discovery**: Automatic plugin discovery via entry points
- **Lazy Loading**: Plugins are loaded on-demand for better performance
- **Dependency Resolution**: Automatic topological sorting with circular dependency detection
- **Thread-Safe**: Concurrent access protection with locks
- **Flexible Filtering**: Advanced query capabilities for component discovery

## Architecture

The registry uses a singleton pattern similar to Django's app registry but with better plugin support:

```
ComponentRegistry (Singleton)
├── core namespace (built-in components)
│   ├── nodes
│   ├── adapters
│   ├── tools
│   ├── policies
│   ├── memory
│   └── observers
├── plugin_a namespace
│   └── custom components
└── plugin_b namespace
    └── specialized components
```

## Component Types

The registry supports the following component types via the `ComponentType` enum:

- `ComponentType.NODE`: DAG execution nodes
- `ComponentType.ADAPTER`: Port implementations
- `ComponentType.TOOL`: Agent tools and utilities
- `ComponentType.POLICY`: Pipeline policies
- `ComponentType.MEMORY`: Memory storage components
- `ComponentType.OBSERVER`: Event observers

Both enum values and string literals are supported for backward compatibility.

## Namespaces

The registry uses namespaces to organize components and prevent conflicts.

### Protected Namespaces
These namespaces are reserved for framework use and can only be used by `hexai.core.*` modules:

- **`core`**: Essential framework components (always available)
  - Example: `PassThroughNode`, `LoggingNode`
- **`hexai`**: Official optional extensions and integrations
  - Example: Cloud adapters, premium features
- **`system`**: Runtime system components and monitoring
  - Example: System monitors, profilers
- **`internal`**: Private implementation details (not for direct use)
  - Example: Internal serializers, helpers

### User Namespaces
Plugins and user code must use their own namespaces:

```python
# Good - uses plugin's namespace
@node(namespace='my_plugin')
class CustomNode:
    pass

# Bad - will raise ValueError
@node(namespace='core')
class MyNode:
    pass
```

The protection is enforced by checking the calling module - only `hexai.core.*` modules can register in protected namespaces.

## Basic Usage

### Registering Components

```python
from hexai.core.registry import registry, ComponentMetadata, ComponentType

# Register a simple component
registry.register(
    name='llm_processor',
    component=LLMProcessorNode(),
    component_type=ComponentType.NODE
)

# Register with full metadata
registry.register(
    name='advanced_processor',
    component=AdvancedProcessor(),
    component_type=ComponentType.NODE,
    metadata=ComponentMetadata(
        name='advanced_processor',
        component_type=ComponentType.NODE,
        version='2.0.0',
        description='Advanced text processor with LLM',
        tags=frozenset({'llm', 'text', 'ai'}),
        author='hexdag-team',
        dependencies=frozenset({'tokenizer', 'llm_adapter'})
    )
)

# Register with kwargs (metadata created automatically)
registry.register(
    name='memory_store',
    component=MemoryStore(),
    component_type=ComponentType.MEMORY,
    version='1.0.0',
    description='In-memory storage for agent state',
    tags=frozenset({'memory', 'storage'})
)
```

### Retrieving Components

```python
# Get a specific component
node = registry.get('llm_processor', component_type=ComponentType.NODE)

# Get from plugin namespace
plugin_tool = registry.get('my_plugin:custom_tool', component_type=ComponentType.TOOL)

# Search across all types
component = registry.get('some_component')  # Searches all types

# Get component metadata
metadata = registry.get_metadata('llm_processor', component_type=ComponentType.NODE)
```

### Component Discovery

```python
# List all nodes
nodes = registry.list_components(component_type=ComponentType.NODE)

# List plugin components
plugin_components = registry.list_components(namespace='my_plugin')

# List with metadata
components_with_meta = registry.list_components(include_metadata=True)

# Find components by filters
llm_components = registry.find(tags__contains='llm')
v2_components = registry.find(version__startswith='2.0')
team_components = registry.find(author='hexdag-team')

# Complex queries
ai_nodes = registry.find(
    component_type=ComponentType.NODE,
    tags__contains='ai',
    version__startswith='2.'
)
```

### Dependency Resolution

```python
# Register components with dependencies
registry.register(
    name='data_processor',
    component=DataProcessor(),
    component_type=ComponentType.NODE,
    dependencies=frozenset({'database', 'cache'})
)

# Resolve dependencies (returns topological order)
init_order = registry.resolve_dependencies('data_processor')
# Returns: ['database', 'cache', 'data_processor']

# Initialize components in correct order
for component_name in init_order:
    component = registry.get(component_name)
    if hasattr(component, 'initialize'):
        component.initialize()
```

## Plugin Development

### Creating a Plugin

Plugins extend hexDAG by registering components in their own namespace:

```python
# my_plugin/__init__.py

from hexai.core.registry import ComponentMetadata

class CustomAnalyzer:
    """Custom analyzer from plugin."""

    def execute(self, data):
        return f"Analyzed: {data}"

def register(registry, namespace):
    """Register plugin components.

    This function is called by hexDAG when the plugin loads.

    Parameters
    ----------
    registry : ComponentRegistry
        The hexDAG registry instance.
    namespace : str
        Your plugin's assigned namespace.
    """
    # Register your components
    registry.register(
        name='custom_analyzer',
        component=CustomAnalyzer(),
        component_type='node',
        namespace=namespace,
        metadata=ComponentMetadata(
            name='custom_analyzer',
            component_type='node',
            version='1.0.0',
            description='Custom analyzer node',
            tags=frozenset({'analyzer', 'custom'}),
            author='my-plugin'
        )
    )
```

### Plugin Entry Points

Configure your plugin's entry point in `setup.py`:

```python
setup(
    name='hexdag-my-plugin',
    # ... other setup config ...
    entry_points={
        'hexdag.plugins': [
            'my_plugin = my_plugin:register'
        ]
    }
)
```

Or in `pyproject.toml`:

```toml
[project.entry-points."hexdag.plugins"]
my_plugin = "my_plugin:register"
```

### Loading Plugins

```python
from hexai.core.registry import registry

# Load all installed plugins
plugins_loaded = registry.load_plugins()
print(f"Loaded {plugins_loaded} plugins")

# Now use plugin components
analyzer = registry.get('my_plugin:custom_analyzer', 'node')
result = analyzer.execute("test data")
```

### Manual Plugin Registration

For development or testing:

```python
# Register namespace manually
registry.register_namespace('my_plugin')

# Register components
registry.register(
    name='test_component',
    component=TestComponent(),
    component_type='node',
    namespace='my_plugin'
)

# List plugin namespaces
namespaces = registry.list_namespaces()
print(f"Active namespaces: {namespaces}")
```

## Advanced Features

### Registration Hooks

Add validation or logging to component registration:

```python
def validate_node(name, component, metadata, namespace):
    """Ensure all nodes have execute method."""
    if metadata.component_type == 'node':
        if not hasattr(component, 'execute'):
            raise ValueError(f"Node {name} must have execute method")

# Add pre-registration hook
registry.add_hook(validate_node, 'pre')

# Add post-registration hook for logging
def log_registration(name, component, metadata, namespace):
    print(f"Registered {namespace}:{name} ({metadata.component_type})")

registry.add_hook(log_registration, 'post')
```

### Lazy Loading

Plugins are discovered at startup but loaded only when accessed:

```python
# Check available plugins without loading them
plugins = registry.list_available_plugins()
for namespace, is_loaded in plugins.items():
    print(f"{namespace}: {'loaded' if is_loaded else 'available'}")

# Plugin is automatically loaded when first accessed
component = registry.get('analyzer', namespace='nlp_tools')  # Loads nlp_tools plugin

# Or explicitly load a plugin
if registry.load_plugin('nlp_tools'):
    print("NLP tools plugin loaded")

# List components without loading plugins
components = registry.list_components(include_unloaded=True)
```

### Filter Operations

The registry supports advanced filtering:

```python
# Contains: Check if value is in collection
registry.find(tags__contains='llm')

# Startswith: String prefix matching
registry.find(version__startswith='2.0')

# In: Check if field value is in provided collection
registry.find(component_type__in=['node', 'adapter'])

# Combine filters
registry.find(
    author='hexdag',
    tags__contains='ai',
    component_type='node'
)
```

### Component Replacement and Protection

The registry follows an append-only pattern with controlled replacement:

```python
# Mark component as replaceable during registration
registry.register(
    name='dev_component',
    component=DevComponent(),
    component_type=ComponentType.NODE,
    replaceable=True  # Explicitly allow future replacement
)

# Later, replace the component
registry.register(
    name='dev_component',
    component=ImprovedDevComponent(),
    component_type=ComponentType.NODE,
    replace=True  # Works because replaceable=True
)

# Protected components (default behavior)
registry.register(
    name='core_component',
    component=CoreComponent(),
    component_type=ComponentType.NODE,
    replaceable=False  # Default - cannot be replaced
)

# Dependency protection
registry.register(
    name='database',
    component=Database(),
    component_type=ComponentType.ADAPTER,
    replaceable=True
)

registry.register(
    name='service',
    component=Service(),
    component_type=ComponentType.NODE,
    dependencies={'database'}  # Depends on database
)

# Now database cannot be replaced (has dependents)
try:
    registry.register('database', NewDatabase(), replace=True)
except ValueError:
    # "Component 'core:database' cannot be replaced because it has dependents"
    pass

# Check dependents before replacement
dependents = registry.get_dependents('database')
print(f"Components depending on database: {dependents}")

# Emergency override (use with extreme caution)
registry.register(
    name='database',
    component=NewDatabase(),
    component_type=ComponentType.ADAPTER,
    replace=True,
    force_replace=True  # Force replacement despite dependents
)
```

### Namespace Management

```python
# List all namespaces
namespaces = registry.list_namespaces()

# Register new namespace for plugin
registry.register_namespace('my_plugin')

# Note: Namespaces and components cannot be removed in production
# The registry follows an append-only pattern for stability
```

## Component Metadata Schema

```python
class ComponentMetadata(BaseModel):
    """Component metadata with full validation."""

    model_config = ConfigDict(frozen=True)  # Immutable

    name: str                                    # Unique identifier
    component_type: ComponentType | str          # Component type enum or string
    version: str | None = None                   # Semantic versioning
    description: str | None = None               # Human-readable description
    tags: frozenset[str] = Field(                # Categorization tags
        default_factory=frozenset
    )
    author: str = Field(default='hexdag')        # Component author
    dependencies: frozenset[str] = Field(        # Required components
        default_factory=frozenset
    )
    config_schema: type[BaseModel] | None = None # Pydantic config model
    replaceable: bool = False                    # Whether component can be replaced
```

## Configuration Validation

Use Pydantic models for component configuration:

```python
from pydantic import BaseModel, Field

class LLMConfig(BaseModel):
    """Configuration for LLM components."""
    model: str = "gpt-4"
    temperature: float = Field(0.7, ge=0, le=2)
    max_tokens: int = Field(1000, ge=1, le=10000)

# Register with config schema
registry.register(
    name='llm_node',
    component=LLMNode,
    component_type='node',
    metadata=ComponentMetadata(
        name='llm_node',
        component_type='node',
        config_schema=LLMConfig
    )
)

# Use config schema
metadata = registry.get_metadata('llm_node')
if metadata.config_schema:
    config = metadata.config_schema(
        model="gpt-4",
        temperature=0.8,
        max_tokens=2000
    )
    node = registry.get('llm_node')
    initialized_node = node(config) if callable(node) else node
```

## Best Practices

### 1. Use Namespaces for Plugins

Always register plugin components in their own namespace:

```python
# Good: Plugin uses its namespace
registry.register('analyzer', Analyzer(), ComponentType.NODE, namespace='my_plugin')

# Bad: Plugin pollutes core namespace
registry.register('analyzer', Analyzer(), ComponentType.NODE)  # Goes to 'core'
```

### 2. Provide Complete Metadata

Include comprehensive metadata for discoverability:

```python
registry.register(
    name='component',
    component=Component(),
    component_type='node',
    metadata=ComponentMetadata(
        name='component',
        component_type='node',
        version='1.2.3',
        description='Clear description of what this does',
        tags=frozenset({'relevant', 'tags', 'for', 'search'}),
        author='your-name',
        dependencies=frozenset({'required_component_names'})
    )
)
```

### 3. Use Semantic Versioning

Follow semantic versioning for components:

```python
version="MAJOR.MINOR.PATCH"
# MAJOR: Breaking changes
# MINOR: New features, backward compatible
# PATCH: Bug fixes
```

### 4. Document Dependencies

Clearly document why dependencies are needed:

```python
ComponentMetadata(
    name="report_generator",
    component_type="node",
    dependencies=frozenset({
        'data_analyzer',     # Provides analyzed data
        'template_engine',   # Renders report templates
        'pdf_converter'      # Converts HTML to PDF
    })
)
```

### 5. Handle Component Types Properly

Components can be registered as classes or instances:

```python
# Register as class (instantiated when retrieved)
registry.register('node_class', NodeClass, ComponentType.NODE)

# Register as instance (singleton)
registry.register('node_instance', NodeClass(), ComponentType.NODE)

# Handle both when retrieving
component = registry.get('some_node', ComponentType.NODE)
if isinstance(component, type):
    # It's a class, instantiate it
    instance = component()
else:
    # It's already an instance
    instance = component
```

## Integration with hexDAG

### With DAG Orchestration

```python
from hexai.core.registry import registry

# Build DAG from registered components
nodes = registry.list_components(component_type='node')

for node_name in nodes:
    metadata = registry.get_metadata(node_name)
    node = registry.get(node_name, 'node')

    # Add to DAG with dependencies
    dag.add_node(node_name, node)
    for dep in metadata.dependencies:
        dag.add_edge(dep, node_name)
```

### With Event System

```python
# Register observer components
registry.register(
    'metrics_observer',
    MetricsObserver(),
    ComponentType.OBSERVER,
    description='Collects execution metrics'
)

# Discover and attach observers
observers = registry.list_components(component_type=ComponentType.OBSERVER)
for observer_name in observers:
    observer = registry.get(observer_name, ComponentType.OBSERVER)
    event_bus.attach(observer)
```

### With Agent Factory

```python
# Register agent tools
registry.register(
    'web_search_tool',
    WebSearchTool(),
    ComponentType.TOOL,
    tags=frozenset({'agent', 'search', 'web'})
)

# Agent discovers available tools
agent_tools = registry.find(
    component_type=ComponentType.TOOL,
    tags__contains='agent'
)
```

## Testing

### Testing with the Registry

```python
import pytest
from hexai.core.registry import registry

@pytest.fixture
def clean_registry():
    """Provide clean registry for tests."""
    # Clear non-core components
    registry.clear()

    # Register test namespace
    if 'test' not in registry.list_namespaces():
        registry.register_namespace('test')

    yield registry

    # Cleanup
    registry.clear(namespace='test')

def test_component_registration(clean_registry):
    """Test registering and retrieving components."""
    # Register test component
    clean_registry.register(
        'test_node',
        TestNode(),
        'node',
        namespace='test'
    )

    # Verify registration
    node = clean_registry.get('test:test_node', 'node')
    assert node is not None

    # Verify in listings
    test_components = clean_registry.list_components(namespace='test')
    assert 'test:test_node' in test_components
```

### Mocking Components

```python
def test_with_mock_component(clean_registry):
    """Test with mocked component."""
    mock_llm = Mock(spec=['generate'])
    mock_llm.generate.return_value = "Mocked response"

    # Replace real component with mock
    clean_registry.register(
        'llm_adapter',
        mock_llm,
        ComponentType.ADAPTER,
        namespace='test',
        replace=True
    )

    # Test uses mock
    llm = clean_registry.get('test:llm_adapter', 'adapter')
    result = llm.generate("test prompt")
    assert result == "Mocked response"
```

## Troubleshooting

### Component Not Found

```python
try:
    component = registry.get('missing_component', 'node')
except KeyError as e:
    print(f"Error: {e}")

    # List available components
    available = registry.list_components(component_type='node')
    print(f"Available nodes: {available}")
```

### Circular Dependencies

```python
try:
    order = registry.resolve_dependencies('component_with_circular_dep')
except ValueError as e:
    print(f"Circular dependency detected: {e}")

    # Debug dependencies
    metadata = registry.get_metadata('component_with_circular_dep')
    print(f"Dependencies: {metadata.dependencies}")
```

### Plugin Not Loading

```python
# Check if plugin is installed
from importlib.metadata import entry_points

eps = entry_points(group='hexdag.plugins')
print(f"Found plugins: {[ep.name for ep in eps]}")

# Try manual loading
from my_plugin import register
registry.register_namespace('my_plugin')
register(registry, 'my_plugin')
```

### Namespace Conflicts

```python
# Check existing namespaces
namespaces = registry.list_namespaces()
print(f"Registered namespaces: {namespaces}")

# Use unique namespace for your plugin
if 'my_namespace' not in namespaces:
    registry.register_namespace('my_namespace')
```

## Performance Considerations

- **Singleton Pattern**: Single registry instance eliminates overhead
- **Thread-Safe**: Lock-protected for concurrent access
- **Lazy Loading**: Components loaded on-demand
- **Efficient Filtering**: Direct dictionary access for lookups
- **Namespace Isolation**: O(1) namespace checks

## Migration Guide

### From Custom Registries

If migrating from the previous multi-registry design:

```python
# Old approach (no longer supported)
# class MyRegistry(BaseRegistry):
#     pass

# New approach
from hexai.core.registry import registry

# Register in a namespace instead
registry.register_namespace('my_feature')
registry.register(
    'component',
    Component(),
    'node',
    namespace='my_feature'
)
```

### From Direct Imports

```python
# Old: Direct imports
from my_module import MyNode
node = MyNode()

# New: Registry-based
from hexai.core.registry import registry

registry.register('my_node', MyNode(), 'node')
node = registry.get('my_node', 'node')
```

## API Reference

### ComponentRegistry

The main registry class (singleton, append-only):

- `register()` - Register a component (with optional `replaceable` flag)
- `get()` - Retrieve a component
- `get_metadata()` - Get component metadata
- `list_components()` - List registered components
- `find()` - Find components by filters
- `resolve_dependencies()` - Get initialization order
- `get_dependents()` - Get components that depend on a given component
- `register_namespace()` - Register plugin namespace
- `list_namespaces()` - List all namespaces
- `add_hook()` - Add registration hook
- `remove_hook()` - Remove registration hook
- `load_plugins()` - Load plugins from entry points
- `_clear_for_testing()` - Clear components (testing only, requires test environment)

### ComponentMetadata

Pydantic model for component metadata:

- `name` - Component identifier
- `component_type` - Type (ComponentType enum or string)
- `version` - Semantic version
- `description` - Human-readable description
- `tags` - Search/filter tags
- `author` - Component author
- `dependencies` - Required components
- `config_schema` - Pydantic config model
- `replaceable` - Whether component can be replaced (default: False)

## See Also

- [hexDAG Philosophy](../../../docs/PHILOSOPHY.md) - Core design principles
- [Plugin Example](../../../examples/plugin_example.py) - Complete plugin example
- [Agent Factory](../../agent_factory/README.md) - Agent management system
- [Event System](../application/events/README.md) - Event-driven architecture
