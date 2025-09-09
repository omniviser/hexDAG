# Claude Development Guidelines

## Modern Python 3.12+ Type Hints

This project enforces modern Python 3.12+ type hint syntax and prohibits legacy typing constructs.

### ✅ DO USE (Modern Python 3.12+)
```python
# Built-in generics (Python 3.9+)
def process_items(items: list[str]) -> dict[str, int]: ...
def get_values() -> set[int]: ...
def create_mapping() -> dict[str, list[int]]: ...

# Union types with pipe operator (Python 3.10+)
def parse_value(val: str | int | float) -> str: ...
def find_user(id: int) -> User | None: ...

# Type alias with 'type' statement (Python 3.12+)
type UserId = int
type UserDict = dict[str, User]
```

### ❌ DON'T USE (Legacy)
```python
# OLD: typing module imports
from typing import Dict, List, Set, Optional, Union

# OLD: Capitalized generic types
def process_items(items: List[str]) -> Dict[str, int]: ...

# OLD: Union and Optional
def parse_value(val: Union[str, int, float]) -> str: ...
def find_user(id: int) -> Optional[User]: ...

# OLD: TypeAlias
from typing import TypeAlias
UserId: TypeAlias = int
```

### Enforcement Tools

#### 1. Pyupgrade
- **Purpose**: Automatically upgrades Python syntax to use modern patterns
- **Configuration**: `--py312-plus` flag ensures Python 3.12+ syntax
- **Pre-commit**: Runs automatically to upgrade legacy type hints

#### 2. Ruff
- **Purpose**: Fast Python linter that enforces modern type hints
- **Key Rules**:
  - `UP006`: Use `list` instead of `List`
  - `UP007`: Use `X | Y` instead of `Union[X, Y]`
  - `UP035`: Use `dict` instead of `Dict`
  - `UP037`: Use `X | None` instead of `Optional[X]`
  - `UP040`: Use `type` alias syntax instead of `TypeAlias`

## Type Checking

This project uses modern Python 3.12+ type checkers to ensure code quality and type safety:

### Pyright
- **Description**: Fast, feature-rich type checker developed by Microsoft
- **Python Support**: Full Python 3.12+ compatibility including latest typing features
- **Configuration**: Runs in both pre-commit hooks and Azure pipelines
- **Command**: `uv run pyright ./hexai`
- **Key Features**:
  - Excellent performance and speed
  - Rich VS Code integration (Pylance)
  - Comprehensive type inference
  - Support for advanced typing patterns (TypeVars, Generics, Protocols)

### MyPy
- **Description**: Standard Python type checker with extensive ecosystem support
- **Configuration**: Configured with pydantic and types-PyYAML support
- **Command**: `uv run mypy ./hexai`

## Running Type Checks

### Local Development
```bash
# Run all pre-commit hooks including type checkers
uv run pre-commit run --all-files

# Run Pyright specifically
uv run pyright ./hexai

# Run MyPy specifically
uv run mypy ./hexai
```

### Pre-commit Integration
Both type checkers automatically run on:
- Every commit (pre-commit stage)
- Can be manually triggered with `uv run pre-commit run pyright --all-files`

### CI/CD Pipeline
Type checking is enforced in Azure DevOps pipelines:
- Runs on all pull requests to main branch
- Blocks merge if type errors are detected
- Provides detailed error reporting

## Type Checking Best Practices

1. **Always add type hints** to function signatures and class attributes
2. **Use modern typing features** from Python 3.12+ (e.g., `type` statement, improved generics)
3. **Leverage Pydantic models** for data validation with automatic type inference
4. **Fix type errors immediately** - don't use `# type: ignore` unless absolutely necessary
5. **Run type checkers locally** before committing to catch issues early

## Excluded Paths
The following paths are excluded from type checking:
- `tests/` - Test files often use dynamic mocking that confuses type checkers
- `examples/` - Example code may intentionally show various usage patterns

## Troubleshooting

If you encounter type checking issues:
1. Ensure you're using Python 3.12+: `python --version`
2. Update dependencies: `uv sync`
3. Clear pyright cache: `rm -rf .pyright`
4. Check for conflicting type stubs: `uv pip list | grep types-`
