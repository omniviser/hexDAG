# Plugin Development

## Quick Start

```bash
# Create plugin
hexdag plugin new redis_adapter --port memory

# Develop
hexdag plugin lint redis_adapter
hexdag plugin test redis_adapter

# Install
hexdag plugin install redis_adapter
```

## How It Works

Plugins live in `hexai_plugins/` with:
- Own dependencies (`pyproject.toml`)
- Relaxed linting (configured in main `pyproject.toml`)
- CLI management (`hexdag plugin`)

## Structure

```
hexai_plugins/my_adapter/
├── my_adapter.py        # Implement @adapter decorator
├── pyproject.toml       # Dependencies
└── tests/
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `new <name>` | Create plugin from template |
| `list` | Show all plugins |
| `lint <name>` | Lint with auto-fix |
| `format <name>` | Format code |
| `test <name>` | Run tests |
| `install <name>` | Install plugin |
| `check-all` | Check all plugins |

## Configuration

```toml
# hexdag.toml or pyproject.toml
[tool.hexdag]
plugins = ["hexai_plugins.mysql_adapter"]
```
