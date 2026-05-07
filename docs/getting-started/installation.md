# Installation

## Using uv (Recommended)

```bash
# Install dependencies
uv sync

# Install with notebook support
uv sync --all-extras

# Install pre-commit hooks
uv run pre-commit install
```

## Verify Installation

```bash
# Run tests
uv run pytest

# Check type safety
uv run pyright ./hexdag

# Run linting
uv run ruff check hexdag/
```

## Development Setup

```bash
# Run all pre-commit hooks
uv run pre-commit run --all-files

# Start Jupyter for interactive development
jupyter notebook notebooks/
```
