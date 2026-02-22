# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

hexDAG is an **operating system for AI agents** -- an orchestration framework that provides pipelines (processes), ports (syscalls), drivers, and a standard library so that AI agents don't reinvent orchestration. It transforms complex AI workflows into deterministic, testable, and maintainable systems through declarative YAML configurations and DAG-based execution.

## Development Commands

### Environment Setup
```bash
# Install dependencies using uv (Python package manager)
uv sync

# Install with notebook support (for documentation)
uv sync --all-extras

# Install pre-commit hooks
uv run pre-commit install
```

### Notebooks
```bash
# Start Jupyter for interactive development
jupyter notebook notebooks/

# Execute and validate all notebooks
uv run python scripts/check_notebooks.py

# Format notebooks
uv run nbqa ruff notebooks/ --fix
uv run nbqa pyupgrade notebooks/ --py312-plus

# Strip notebook outputs (automatic via pre-commit)
uv run nbstripout notebooks/**/*.ipynb
```

### Testing
```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=hexdag --cov-report=html --cov-report=term-missing

# Run specific test areas
uv run pytest tests/hexdag/kernel/pipeline_builder/ -x --tb=short  # Pipeline builder tests
uv run pytest tests/hexdag/kernel/                                 # Kernel tests
uv run pytest tests/hexdag/stdlib/lib/                             # System lib tests

# Run doctests (tests embedded in docstrings)
uv run pytest --doctest-modules hexdag/ --ignore=hexdag/cli/
uv run pytest --doctest-modules hexdag/ --ignore=hexdag/cli/ --doctest-continue-on-failure  # See all failures
```

### Code Quality
```bash
# Run all pre-commit hooks
uv run pre-commit run --all-files

# Linting and formatting
uv run ruff check hexdag/ --fix         # Linting with auto-fix
uv run ruff format hexdag/              # Code formatting
uv run mypy hexdag/                     # Type checking
uv run pyright hexdag/                  # Alternative type checker
uv run bandit -r hexdag                 # Security scanning

# Dependency analysis
uv run deptry .                        # Unused dependencies
uv run safety check                    # Vulnerability scanning
./scripts/check_licenses.sh            # License compliance

# Code quality metrics
uv run vulture hexdag/ --min-confidence 90    # Dead code detection
uv run radon cc hexdag/ --min B               # Complexity analysis

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

hexDAG is structured like an operating system -- kernel (execution engine), stdlib (built-in components), drivers (infrastructure), and api (user-facing tools):

### Framework Structure
```
hexdag/
├── kernel/                  # Core execution engine (/kernel)
│   ├── domain/              #   Domain models (DAG, NodeSpec, PipelineRun, etc.)
│   ├── orchestration/       #   Orchestrator, events, observers
│   ├── pipeline_builder/    #   YAML pipeline building and compilation
│   ├── ports/               #   Port interfaces (LLM, DataStore, PipelineSpawner)
│   ├── validation/          #   Type validation and schema conversion
│   └── lib_base.py          #   HexDAGLib base class (lib contract)
├── stdlib/                  # Standard library (/lib)
│   ├── adapters/            #   Built-in adapters (OpenAI, SQLite, Mock, etc.)
│   ├── nodes/               #   Node factories (LLMNode, AgentNode, etc.)
│   ├── macros/              #   Macro components (ReasoningAgent, etc.)
│   ├── prompts/             #   Prompt templates (tool prompts, etc.)
│   └── lib/                 #   System libs (ProcessRegistry, EntityState, Scheduler)
├── drivers/                 # Low-level infrastructure (/drivers)
│   ├── executors/           #   LocalExecutor (ExecutorPort)
│   ├── observer_manager/    #   LocalObserverManager (ObserverManagerPort)
│   └── pipeline_spawner/    #   LocalPipelineSpawner (PipelineSpawner)
├── api/                     # Unified API layer (/usr/bin)
│   ├── execution.py         #   Pipeline execution
│   ├── processes.py         #   Process management (9 MCP tools)
│   └── ...                  #   Components, validation, documentation
├── docs/                    # Documentation utilities
└── cli/                     # Command-line interface
```

### Uniform Entity Pattern

All framework entities follow one pattern: **kernel defines contract, stdlib ships builtins, users write their own.**

| Entity   | Contract (kernel/)          | Builtins (stdlib/)                 | User custom        |
|----------|----------------------------|------------------------------------|-------------------|
| Ports    | `kernel/ports/llm.py`      | -                                  | `myapp.ports.X`   |
| Adapters | `kernel/ports/` (Protocol) | `stdlib/adapters/openai/`          | `myapp.adapters.X`|
| Nodes    | `stdlib/nodes/base_node_factory.py` | `stdlib/nodes/llm_node.py` | `myapp.nodes.X`   |
| Macros   | (convention)               | `stdlib/macros/reasoning_agent.py` | `myapp.macros.X`  |
| Prompts  | (convention)               | `stdlib/prompts/tool_prompts.py`   | `myapp.prompts.X` |
| **Libs** | **`kernel/lib_base.py`**   | **`stdlib/lib/process_registry.py`** | **`myapp.lib.X`** |

### System Libraries (Libs)

Libs are the new entity type for multi-pipeline coordination:

- **ProcessRegistry** — tracks pipeline runs (like `ps` in Linux)
- **EntityState** — declarative state machines for business entities
- **Scheduler** — delayed/recurring pipeline execution (asyncio timers)
- **DatabaseTools** — agent-callable SQL query tools

Every public async method on a `HexDAGLib` subclass auto-becomes an agent tool.

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
- **Command**: `uv run pyright ./hexdag`
- **Key Features**:
  - Excellent performance and speed
  - Rich VS Code integration (Pylance)
  - Comprehensive type inference
  - Support for advanced typing patterns (TypeVars, Generics, Protocols)

### MyPy
- **Description**: Standard Python type checker with extensive ecosystem support
- **Configuration**: Configured with pydantic and types-PyYAML support
- **Command**: `uv run mypy ./hexdag`

## Running Type Checks

### Local Development
```bash
# Run all pre-commit hooks including type checkers
uv run pre-commit run --all-files

# Run Pyright specifically
uv run pyright ./hexdag

# Run MyPy specifically
uv run mypy ./hexdag
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
2. **Runtime Warnings** - `hexdag/kernel/utils/async_warnings.py` detects blocking operations at runtime

#### Running the Async I/O Checker

```bash
# Check entire codebase
uv run python scripts/check_async_io.py

# Verbose output
uv run python scripts/check_async_io.py --verbose

# Check specific paths
uv run python scripts/check_async_io.py hexdag/drivers/
```

#### Using Runtime Warnings

```python
from hexdag.kernel.utils.async_warnings import warn_sync_io, warn_if_async

# In async functions
async def my_function():
    warn_sync_io("file_open", "Use aiofiles.open()")

# As decorator
@warn_if_async
def sync_helper():
    return open('file.txt').read()
```

See `docs/async_io_enforcement.md` for complete documentation.

### Error Handling
- All framework exceptions inherit from `HexDAGError`
- Use custom exception hierarchies (e.g., `OrchestratorError`, `NodeValidationError`)
- Maintain error context and node information
- Event emission for error tracking

## Component Resolution

hexDAG uses a **module path resolver** for component discovery. Components (nodes, adapters, tools) are referenced by their full Python module path.

### How It Works

```python
from hexdag.kernel.resolver import resolve

# Resolve a node class
LLMNode = resolve("hexdag.stdlib.nodes.LLMNode")

# Resolve an adapter class
MockLLM = resolve("hexdag.stdlib.adapters.mock.MockLLM")

# Resolve a custom component
MyNode = resolve("myapp.nodes.MyNode")
```

### Using in YAML Pipelines

Components are referenced by module path in YAML:

```yaml
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: my-pipeline
spec:
  ports:
    llm:
      adapter: hexdag.stdlib.adapters.openai.OpenAIAdapter
      config:
        model: gpt-4
  nodes:
    - kind: hexdag.stdlib.nodes.LLMNode
      metadata:
        name: analyzer
      spec:
        prompt_template: "Analyze: {{input}}"
```

### Built-in Aliases

For convenience, built-in nodes have short aliases:

```yaml
# These are equivalent:
- kind: llm_node                           # Short alias
- kind: hexdag.stdlib.nodes.LLMNode       # Full path
```

Available aliases: `llm_node`, `function_node`, `agent_node`, `loop_node`, `conditional_node`

### Creating Custom Components

**Adapters** implement port interfaces:

```python
from hexdag.kernel.ports.llm import LLM

class MyLLMAdapter(LLM):
    def __init__(self, model: str = "gpt-4", api_key: str | None = None):
        self.model = model
        self.api_key = api_key

    async def aresponse(self, messages: list[dict]) -> str:
        # Your implementation
        ...
```

**Nodes** extend BaseNodeFactory:

```python
from hexdag.stdlib.nodes import BaseNodeFactory
from hexdag.kernel.domain import NodeSpec

class MyNode(BaseNodeFactory):
    def __call__(self, name: str, config: dict, **kwargs) -> NodeSpec:
        async def process(inputs: dict, context):
            return {"result": "processed"}

        return NodeSpec(id=name, fn=process)
```

**Libs** extend HexDAGLib (async methods auto-become agent tools):

```python
from hexdag.kernel.lib_base import HexDAGLib
from hexdag.kernel.ports.data_store import SupportsKeyValue

class OrderManager(HexDAGLib):
    def __init__(self, store: SupportsKeyValue) -> None:
        self._store = store

    async def acreate_order(self, customer_id: str, items: list[dict]) -> str:
        """Create a new order. Auto-exposed as agent tool."""
        # Your implementation
        ...

    async def aget_order(self, order_id: str) -> dict:
        """Get order by ID. Auto-exposed as agent tool."""
        ...
```

**Tools** are plain functions with type hints:

```python
def search_database(query: str, limit: int = 10) -> list[dict]:
    """Search the database for matching records.

    Args:
        query: Search query string
        limit: Maximum number of results

    Returns:
        List of matching records
    """
    # Your implementation
    return results
```

### Explicit `__init__` Parameters (Convention)

**IMPORTANT:** All adapter/component `__init__` methods MUST use **explicit typed parameters** instead of `**kwargs` for configuration. This enables automatic schema generation via `SchemaGenerator`.

**Why this matters:**
- `SchemaGenerator.from_callable()` introspects `__init__` signatures to generate JSON Schema
- Studio UI, MCP server, and API all use these schemas to show configuration options
- `**kwargs`-only signatures result in **empty schemas** - users can't see what options exist

**✅ CORRECT - Explicit parameters:**
```python
class MockLLM(LLM):
    def __init__(
        self,
        responses: str | list[str] | None = None,
        delay_seconds: float = 0.0,
        mock_tool_calls: list[dict[str, Any]] | None = None,
        **kwargs: Any,  # Keep for forward compatibility
    ) -> None:
        self.responses = responses or ['{"result": "Mock response"}']
        self.delay_seconds = delay_seconds
        self.mock_tool_calls = mock_tool_calls
```

**❌ WRONG - kwargs-only (generates empty schema):**
```python
class MockLLM(LLM):
    def __init__(self, **kwargs: Any) -> None:
        # SchemaGenerator can't see these options!
        self.responses = kwargs.get("responses", ['{"result": "Mock"}'])
        self.delay_seconds = kwargs.get("delay_seconds", 0.0)
```

## Working with YAML Pipelines

YAML pipelines are built using `YamlPipelineBuilder` in `hexdag/kernel/pipeline_builder/`:

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
- **Libs**: System libraries whose methods auto-become agent tools (ProcessRegistry, EntityState, Scheduler)

### Function Nodes with Module Path Strings

Function nodes support fully declarative function references using module path strings:

```yaml
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: data-processing
spec:
  nodes:
    # Standard library functions
    - kind: function_node
      metadata:
        name: json_parser
      spec:
        fn: "json.loads"  # Module path string - no imports needed!
      dependencies: []

    # Your custom business logic
    - kind: function_node
      metadata:
        name: process_order
      spec:
        fn: "myapp.business.process_order"
        input_schema:
          order_id: str
          customer_id: str
        output_schema:
          status: str
          total: float
      dependencies: [json_parser]

    # Third-party packages
    - kind: function_node
      metadata:
        name: data_transform
      spec:
        fn: "pandas.DataFrame.from_dict"
      dependencies: [process_order]
```

**Benefits:**
- **100% Declarative** - No Python imports in YAML files
- **Git-Friendly** - Pure YAML configurations version-controlled
- **Clear Error Messages** - Validation at build time with descriptive errors
- **Universal** - Works with stdlib, third-party packages, and custom code

See [docs/reference/nodes.md](docs/reference/nodes.md#function_node) for complete documentation.

## External Dependencies

The framework integrates with external services through ports:
- **LLM Port**: Language model interactions (OpenAI, Anthropic, etc.)
- **DataStore Port**: Unified data access (`SupportsKeyValue`, `SupportsQuery`, `SupportsTTL`, `SupportsSchema`, `SupportsTransactions`)
- **PipelineSpawner Port**: Fork/exec for child pipelines
- **Tool Router**: Function calling and tool execution

Use mock adapters in `hexdag/stdlib/adapters/mock/` for development and testing.

## YAML-First Philosophy

hexDAG emphasizes a **declarative, YAML-first approach** to workflow orchestration:

### Why YAML-First?

1. **Declarative** - Describe what you want, not how to build it
2. **Version Control** - Git-friendly, reviewable configurations
3. **Team Collaboration** - Non-developers can read and modify workflows
4. **Environment Management** - Easy dev/staging/prod configurations
5. **Infrastructure as Code** - Deploy workflows like infrastructure
6. **Testable** - Validate YAML before execution
7. **Maintainable** - Change workflows without code changes

### YAML-First Development

When creating examples or documentation:
- ✅ **START with YAML** - Show YAML pipeline definition first
- ✅ **Python API is secondary** - Only show for advanced use cases
- ✅ **Emphasize declarative benefits** - Version control, collaboration, maintainability
- ✅ **Save to .yaml files** - Show file-based workflow management
- ✅ **Environment configs** - Demonstrate dev/staging/prod patterns
- ❌ **Avoid Python-first** - Don't start with DirectedGraph code unless necessary

### Example Structure

```python
# ✅ GOOD - YAML-first approach
pipeline_yaml = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: my-workflow
spec:
  nodes:
    - type: llm
      id: analyzer
      # ... config
"""
pipeline = YamlPipelineBuilder().build_from_string(pipeline_yaml)

# ❌ AVOID - Python-first approach (unless teaching internals)
graph = DirectedGraph()
graph.add_node(NodeSpec(id="analyzer", ...))
```

## Notebook Development Guidelines

hexDAG uses Jupyter notebooks for **interactive documentation and real-world use cases**.

### Notebook Structure

All notebooks follow this structure:

```
notebooks/
├── 01_getting_started/       # YAML-first tutorials
├── 02_real_world_use_cases/  # Production-ready examples
└── 03_advanced_patterns/     # Enterprise patterns
```

### Writing Notebooks

**✅ DO:**
- Focus on **real-world business problems** with clear value
- Use **YAML pipelines** as the primary interface
- Include **comprehensive markdown** explaining concepts
- Ensure **end-to-end execution** without errors
- Use **mock adapters** to avoid external dependencies
- Add **visualizations** (DAG graphs, metrics, results)
- Follow the **standard structure** (Overview → Problem → Solution → Implementation → Analysis → Extensions)
- **Strip outputs** before committing (automatic via pre-commit)

**❌ DON'T:**
- Create simple code examples (use integration tests instead)
- Require external API keys when avoidable
- Leave outputs in committed notebooks
- Skip documentation in markdown cells
- Focus on Python API over YAML

### Notebook Categories

1. **Getting Started (01/)** - YAML-first onboarding
   - Introduction to YAML pipelines
   - Component overview
   - Validation and type safety

2. **Real-World Use Cases (02/)** - Production scenarios
   - Customer support automation
   - Document intelligence
   - Research assistants
   - Data pipeline orchestration
   - Code analysis and review

3. **Advanced Patterns (03/)** - Enterprise techniques
   - Multi-agent collaboration (YAML)
   - Dynamic workflows (YAML)
   - Production deployment patterns (YAML)
   - Performance optimization
   - Observability and monitoring

### Notebook Quality Standards

- **Validation**: Execute without errors via `scripts/check_notebooks.py`
- **Formatting**: Auto-formatted via `nbqa-ruff` and `nbqa-pyupgrade`
- **Outputs**: Stripped via `nbstripout` (pre-commit hook)
- **CI/CD**: Executed in Azure pipelines to ensure freshness
- **Structure**: Follow nbformat 4.4 specification

### Creating New Notebooks

```bash
# Create notebook
jupyter notebook notebooks/02_real_world_use_cases/new_use_case.ipynb

# Validate
uv run python scripts/check_notebooks.py

# Format
uv run nbqa ruff notebooks/ --fix

# Commit (outputs automatically stripped)
git add notebooks/
git commit -m "docs: Add new use case notebook"
```
