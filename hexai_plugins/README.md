# HexDAG Plugins

External plugins that extend HexDAG with custom adapters.

## Quick Start

### Create a Plugin
```bash
hexdag plugin new redis_adapter --port memory
```

### Develop
```bash
hexdag plugin lint redis_adapter
hexdag plugin test redis_adapter
hexdag plugin check-all
```

### Install
```bash
hexdag plugin install redis_adapter
```

## How Plugins Work

- **Location**: `hexai_plugins/` directory
- **Structure**: Each plugin has its own folder with `pyproject.toml`
- **Rules**: Relaxed linting/typing via main `pyproject.toml` configurations
- **Testing**: Same pre-commit and CI/CD pipeline as main code

## Available Plugins

| Plugin | Description | Port |
|--------|-------------|------|
| `mysql_adapter` | Production MySQL adapter with JSON support | database |

## Plugin Structure

```
plugin_name/
├── __init__.py
├── plugin_name.py      # Adapter implementation
├── pyproject.toml      # Dependencies & config
├── README.md
├── LICENSE
└── tests/
    └── test_plugin.py
```

## Development

All plugin operations through the CLI:
```bash
hexdag plugin --help
```

See [Plugin Development Guide](../docs/PLUGIN_DEVELOPMENT.md) for details.
