# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

hexDAG is an enterprise-ready AI agent orchestration framework that transforms complex AI workflows into deterministic, testable, and maintainable systems through declarative YAML configurations and DAG-based orchestration.

## Development Commands

### Environment Setup
```bash
# Install dependencies using uv (Python package manager)
uv sync

# Install pre-commit hooks
uv run pre-commit install
```

### Testing
```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=hexai --cov-report=html --cov-report=term-missing

# Run specific test areas
uv run pytest tests/hexai/agent_factory/ -x --tb=short  # Agent factory tests
uv run pytest tests/hexai/core/                        # Core framework tests
uv run pytest tests/hexai/validation/                  # Validation tests

# Run doctests (tests embedded in docstrings)
uv run pytest --doctest-modules hexai/ --ignore=hexai/cli/
uv run pytest --doctest-modules hexai/ --ignore=hexai/cli/ --doctest-continue-on-failure  # See all failures
```

### Code Quality
```bash
# Run all pre-commit hooks
uv run pre-commit run --all-files

# Linting and formatting
uv run ruff check hexai/ --fix         # Linting with auto-fix
uv run ruff format hexai/              # Code formatting
uv run mypy hexai/                     # Type checking
uv run pyright hexai/                  # Alternative type checker
uv run bandit -r hexai                 # Security scanning

# Dependency analysis
uv run deptry .                        # Unused dependencies
uv run safety check                    # Vulnerability scanning
./scripts/check_licenses.sh            # License compliance

# Code quality metrics
uv run vulture hexai/ --min-confidence 90    # Dead code detection
uv run radon cc hexai/ --min B               # Complexity analysis

# Coverage and testing
./scripts/coverage_report.sh           # Full coverage report
./scripts/diff_coverage.sh             # Coverage on changed code only
./scripts/run_mutation_tests.sh        # Mutation testing (slow!)

# API compatibility
./scripts/check_api_compat.sh          # Check for breaking changes

# Memory profiling
./scripts/profile_memory.sh            # Memory leak detection
```

### Examples
```bash
# Run all examples
uv run examples/run_all.py

# Run specific examples
uv run examples/01_basic_dag.py           # DAG fundamentals
uv run examples/10_agent_nodes.py         # AI agents with tools
uv run examples/13_yaml_pipelines.py      # Declarative workflows
uv run examples/19_complex_workflow.py    # Enterprise patterns
```

### Utilities
```bash
# Check test structure consistency
uv run scripts/check_test_structure.py

# Check examples functionality
uv run scripts/check_examples.py
```

## Architecture Overview

hexDAG follows hexagonal architecture with clear separation of concerns:

### Core Framework Structure
```
hexai/
├── core/
│   ├── domain/          # Core business logic (DAG, NodeSpec, DirectedGraph)
│   ├── application/     # Use cases (Orchestrator, NodeFactory)
│   │   ├── nodes/       # Node implementations (LLMNode, AgentNode, etc.)
│   │   ├── events/      # Event system for observability
│   │   └── prompt/      # Prompt templating system
│   ├── ports/           # Interface definitions (LLM, Database, Memory)
│   └── validation/      # Type validation and schema conversion
├── adapters/            # External service implementations
│   ├── mock/           # Mock implementations for testing
├── agent_factory/       # YAML pipeline building and compilation
└── cli/                # Command-line interface
```

### The Six Pillars
1. **Async-First Architecture** - Non-blocking execution for maximum performance
2. **Event-Driven Observability** - Real-time monitoring via comprehensive event system
3. **Pydantic Validation Everywhere** - Type safety at every layer
4. **Hexagonal Architecture** - Clean separation of business logic and infrastructure
5. **Composable Declarative Files** - Complex workflows from simple YAML components
6. **DAG-Based Orchestration** - Intelligent dependency management and parallelization

## Key Concepts

### DirectedGraph and NodeSpec
- `DirectedGraph`: Manages workflow structure and dependencies
- `NodeSpec`: Defines individual processing steps with type validation
- DAG validation ensures acyclic execution paths

### Orchestrator
- Core execution engine that walks DirectedGraphs in topological order
- Handles concurrent execution using asyncio.gather()
- Provides comprehensive error handling and event emission

### Node Types
- `FunctionNode`: Execute Python functions with validation
- `LLMNode`: Language model interactions with prompt templating
- `ReActAgentNode`: ReAct pattern agents with tool access
- `LoopNode`: Iterative processing with custom conditions
- `ConditionalNode`: Conditional execution paths

### Event System
- Comprehensive observability through events (NodeStarted, NodeCompleted, NodeFailed, etc.)
- Event-driven memory and monitoring
- Observer pattern for extensible monitoring

### Validation Framework
- Multi-strategy validation system supporting Pydantic, type checking, and custom converters
- Automatic schema compatibility checking between connected nodes
- Runtime type coercion and validation

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

### Branch Naming Convention
Branch names must match the pattern: `^(ci|dependabot\/pip|docs|experiment|feat|fix|refactor|test)\/[A-Za-z0-9._-]+$`

Examples:
- `feat/yaml-pipeline-builder`
- `fix/validation-error-handling`
- `refactor/orchestrator-performance`

### Testing Approach
- Unit tests for domain logic
- Integration tests for orchestrator workflows
- Mock adapters for external services
- Example-based testing for user workflows

### Async I/O Enforcement

hexDAG enforces async-first architecture through:

1. **Static Analysis** - `scripts/check_async_io.py` scans for blocking I/O in async functions
2. **Runtime Warnings** - `hexai/core/utils/async_warnings.py` detects blocking operations at runtime
3. **Adapter Decorator Integration** - `@adapter` decorator automatically wraps async methods

#### Running the Async I/O Checker

```bash
# Check entire codebase
uv run python scripts/check_async_io.py

# Verbose output
uv run python scripts/check_async_io.py --verbose

# Check specific paths
uv run python scripts/check_async_io.py hexai/adapters/
```

#### Using Runtime Warnings

```python
from hexai.core.utils.async_warnings import warn_sync_io, warn_if_async

# In async functions
async def my_function():
    warn_sync_io("file_open", "Use aiofiles.open()")

# As decorator
@warn_if_async
def sync_helper():
    return open('file.txt').read()
```

#### Adapter Decorator with Async Monitoring

```python
from hexai.core.registry.decorators import adapter

# Default: warnings enabled
@adapter("database", name="sqlite")
class SQLiteAdapter:
    async def aexecute_query(self, sql):
        # This will be monitored for blocking I/O
        pass

# Disable warnings for intentional sync I/O
@adapter("database", name="local_db", warn_sync_io=False)
class LocalDBAdapter:
    async def aexecute_query(self, sql):
        # No warnings - intentional sync I/O
        pass
```

See `docs/async_io_enforcement.md` for complete documentation.

### Error Handling
- Use custom exception hierarchies (e.g., `OrchestratorError`, `ValidationError`)
- Maintain error context and node information
- Event emission for error tracking

## Component Configuration (Simplified Pattern)

### Overview

hexDAG uses a **simplified decorator-based pattern** for component configuration:
- **No Config classes required** - just plain `__init__` signatures
- **Secrets declared in decorator** - explicit and clean
- **Automatic secret resolution** - from environment or memory
- **Type-safe** - leverages Python type hints and Pydantic

### Creating Adapters with Secrets

**✅ CORRECT - Decorator-based (preferred):**
```python
from hexdag.core.registry import adapter

@adapter("llm", name="openai", secrets={"api_key": "OPENAI_API_KEY"})
class OpenAIAdapter:
    """OpenAI adapter with automatic secret resolution."""

    def __init__(
        self,
        api_key: str,  # ← Auto-resolved from OPENAI_API_KEY env var
        model: str = "gpt-4",
        temperature: float = 0.7
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

    async def aresponse(self, messages):
        # Use self.api_key, self.model, etc.
        ...
```

**Multiple secrets:**
```python
@adapter(
    "database",
    name="postgres",
    secrets={
        "username": "DB_USERNAME",
        "password": "DB_PASSWORD"
    }
)
class PostgresAdapter:
    def __init__(
        self,
        username: str,
        password: str,
        host: str = "localhost",
        port: int = 5432
    ):
        self.connection = f"postgresql://{username}:{password}@{host}:{port}"
```

### Creating Adapters without Secrets

**✅ CORRECT:**
```python
@adapter("cache", name="memory_cache")
class MemoryCacheAdapter:
    """Simple adapter - no secrets needed."""

    def __init__(
        self,
        max_size: int = 100,
        ttl: int = 3600
    ):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl

    async def aget(self, key: str):
        return self.cache.get(key)
```

### Creating Nodes

**✅ CORRECT:**
```python
from hexdag.core.registry import node

@node(name="my_node", namespace="core")
class MyNode(BaseNodeFactory):
    """Custom node - no Config class needed."""

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        name: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        **kwargs
    ):
        # Parameters are passed directly from YAML
        ...
```

### Creating Policies

**✅ CORRECT:**
```python
from hexdag.core.registry import policy

@policy(name="retry_policy", description="Retry failed operations")
class RetryPolicy:
    """Simple policy - no Config class needed."""

    def __init__(self, max_retries: int = 3, backoff: float = 1.0):
        self.max_retries = max_retries
        self.backoff = backoff

    async def evaluate(self, context):
        # Use self.max_retries, self.backoff
        ...
```

### Creating Tools

**✅ CORRECT:**
```python
from hexdag.core.registry import tool

@tool(name="my_tool", namespace="custom", description="Does something useful")
def my_tool(input_param: str, threshold: int = 10) -> str:
    """Tool function with typed parameters.

    Args:
        input_param: Input string to process
        threshold: Processing threshold (default: 10)

    Returns:
        Processed result
    """
    # For tools, the function signature IS the configuration
    # Type hints define input schema
    # Docstring provides descriptions
    return f"Processed: {input_param}"
```

### Secret Resolution

Secrets are automatically resolved in this order:
1. **Explicit kwargs** (highest priority)
2. **Environment variables** (from decorator `secrets` mapping)
3. **Memory port** (from orchestrator, with `secret:` prefix)
4. **Error** (if parameter has no default value)

**Example:**
```python
# Set environment variable
os.environ["OPENAI_API_KEY"] = "sk-..."

# Secret auto-resolved from env
adapter = OpenAIAdapter()  # ✅ api_key resolved

# Explicit value overrides env
adapter = OpenAIAdapter(api_key="explicit-key")  # ✅ Uses explicit value
```

### Benefits of the Simplified Pattern

1. **70% Less Code** - From 13 lines to 4 lines per adapter
2. **Explicit Configuration** - Secrets declared in decorator, visible at a glance
3. **Clean Signatures** - `__init__` uses plain typed parameters
4. **Type Safety** - Leverages Python type hints and runtime validation
5. **Easy Testing** - Direct parameter passing, no Config mocking needed
6. **YAML Compatible** - Full support for declarative pipelines

### Validation Rules

✅ **All new components MUST:**

**Adapters with Secrets:**
- Use `secrets` parameter in `@adapter` decorator
- Declare secret mapping: `secrets={"param_name": "ENV_VAR"}`
- Clean `__init__` signatures with typed parameters

**Adapters without Secrets:**
- Simple `@adapter` decorator without `secrets`
- Plain `__init__` with typed parameters and defaults

**Nodes:**
- Use `@node` decorator
- Parameters passed directly from YAML to `__call__` method

**Policies:**
- Use `@policy` decorator
- Simple `__init__` with typed parameters

**Tools:**
- Use typed function signatures
- Include comprehensive docstrings
- Type hints for automatic schema generation

### Legacy Code

Some existing adapters use the old `ConfigurableAdapter` pattern with Config classes. These still work but are deprecated. New code should use the simplified decorator pattern. See [SIMPLIFIED_PATTERN.md](docs/SIMPLIFIED_PATTERN.md) for migration guide.

## Working with YAML Pipelines

YAML pipelines are built using `YamlPipelineBuilder` in `hexai/agent_factory/`:

```yaml
name: example_workflow
description: AI-powered workflow

nodes:
  - type: agent
    id: researcher
    params:
      initial_prompt_template: "Research: {{topic}}"
      max_steps: 5
    depends_on: []

  - type: llm
    id: analyzer
    params:
      prompt_template: "Analyze: {{researcher.results}}"
    depends_on: [researcher]
```

Key components:
- **Node Types**: agent, llm, function, conditional, loop
- **Dependencies**: Explicit via `depends_on` array
- **Parameters**: Node-specific configuration
- **Template System**: Jinja2-style templating for dynamic content

## External Dependencies

The framework integrates with external services through ports:
- **LLM Port**: Language model interactions (OpenAI, Anthropic, etc.)
- **Memory Port**: Persistent memory for agents
- **Database Port**: Data persistence and retrieval
- **Tool Router**: Function calling and tool execution

Use mock adapters in `hexai/adapters/mock/` for development and testing.
