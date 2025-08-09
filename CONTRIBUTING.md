# Contributing to hexDAG

Thank you for your interest in contributing to hexDAG! This document provides guidelines for contributing to the project.

## Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://dev.azure.com/omniviser/_git/hexDAG
   cd hexdag
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```
   Note: in uv this is optional `run` installs dependencies as well.

3. **Verify setup**
   ```bash
   uv run pytest
   uv run pre-commit run --all-files
   ```

## Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests and quality checks**
   ```bash
   uv run pytest
   uv run pre-commit run --all-files
   ```

4. **Commit and push**
   ```bash
   git add .
   git commit -m "feat: your descriptive commit message"
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request**

## Code Standards

- **Python 3.12+** syntax required
- **Type hints** for all public APIs
- **Docstrings** following NumPy convention
- **Test coverage** for new functionality
- **Pre-commit hooks** must pass

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=hexai

# Run specific test files
uv run pytest tests/unit/hexai/
```

## Documentation

- Update docstrings for any new functions/classes
- Add examples to `examples/` directory
- Update README.md if adding new features

## Questions?

Feel free to open an issue for any questions or suggestions!
