# Installation

## Requirements

- Python 3.12 or higher
- pip or uv package manager

## Basic Installation

Install HexDAG using pip:

```bash
pip install hexdag
```

Or using uv (recommended for development):

```bash
uv pip install hexdag
```

## Installation with Optional Dependencies

### CLI Tools

For command-line interface support:

```bash
pip install hexdag[cli]
```

### Development Tools

For development with testing and linting:

```bash
pip install hexdag[dev]
```

### All Optional Dependencies

To install everything:

```bash
pip install hexdag[cli,dev]
```

## Verify Installation

Check that HexDAG is installed correctly:

```bash
python -c "from hexai.core.domain import DirectedGraph; print('HexDAG installed successfully')"
```

With CLI tools:

```bash
hexdag --version
```

## Next Steps

- [Quick Start](quickstart.md) - Build your first workflow
- [Core Concepts](concepts.md) - Understand HexDAG architecture
