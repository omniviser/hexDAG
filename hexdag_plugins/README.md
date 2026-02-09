# hexDAG Plugins

This directory contains plugins that extend hexDAG's functionality.

## Available Plugins

| Plugin | Description | Port |
|--------|-------------|------|
| `mysql_adapter` | Production MySQL adapter with JSON support | database |

## Simple Plugin Workflow

Each plugin is a standalone Python package. Just use standard Python tools:

### Install & Test

```bash
# Install a plugin
uv pip install -e hexdag_plugins/mysql_adapter/

# Run tests
uv run pytest hexdag_plugins/mysql_adapter/tests/

# Format code (uses settings from plugin's pyproject.toml)
cd hexdag_plugins/mysql_adapter
uv run ruff format .
uv run ruff check --fix .
```

### Pre-commit Integration

Plugin files are automatically handled by the main pre-commit configuration. When you commit:
- **Plugin files** (`hexdag_plugins/`) use plugin-specific hooks with relaxed rules
- **Main code** (`hexdag/`) uses strict project hooks

This happens automatically - just commit normally and the right hooks run!

### Plugin Structure

```
hexdag_plugins/
└── my_plugin/
    ├── pyproject.toml      # Plugin configuration & dependencies
    ├── __init__.py
    ├── my_plugin.py        # Implementation
    └── tests/
        └── test_my_plugin.py
```

### Creating a New Plugin

1. Copy an existing plugin directory as a template
2. Update `pyproject.toml` with your plugin details
3. Implement the required port interface
4. Write tests

The `pyproject.toml` contains all configuration:
- Dependencies
- Tool settings (ruff, mypy, pytest)
- Plugin metadata for hexDAG

## Using Plugins

Plugins are used by referencing their full module path in YAML:

```yaml
spec:
  ports:
    database:
      adapter: hexdag_plugins.mysql_adapter.MySQLAdapter
```

Or import directly in Python:

```python
from hexdag_plugins.mysql_adapter import MySQLAdapter

adapter = MySQLAdapter(host="localhost", database="mydb")
```

That's it! No complex scripts or build tools needed.
