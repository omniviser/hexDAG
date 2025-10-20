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
```

### Code Quality
```bash
# Run all pre-commit hooks
uv run pre-commit run --all-files

# Individual tools
uv run ruff check hexai/ --fix         # Linting with auto-fix
uv run ruff format hexai/              # Ruff formatting
uv run mypy hexai/                     # Type checking
```

### Examples
```bash
# Run specific examples
uv run examples/01_basic_dag.py           # DAG fundamentals
uv run examples/10_agent_nodes.py         # AI agents with tools
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

# Claude Development Guidelines

## Modern Python 3.12+ Type Hints

This project enforces modern Python 3.12+ type hint syntax and prohibits legacy typing constructs.

### ✅ DO USE (Modern Python 3.12+)
```python
# Built-in generics (Python 3.9+)
def process_items(items: list[str]) -> dict[str, int]: ...

# Union types with pipe operator (Python 3.10+)
def find_user(id: int) -> User | None: ...

# Type alias with 'type' statement (Python 3.12+)
type UserId = int
```

### ❌ DON'T USE (Legacy)
```python
# OLD: typing module imports
from typing import Dict, List, Set, Optional, Union

# OLD: Capitalized generic types
def process_items(items: List[str]) -> Dict[str, int]: ...
```

### Enforcement Tools
- **Pyupgrade**: Automatically upgrades Python syntax (`--py312-plus` flag)
- **Ruff**: Enforces modern type hints (UP006, UP007, UP035, UP037, UP040)

_For detailed type hint patterns, see `.claude/type_hints_guide.md`_

## Type Checking
- **Pyright**: Fast type checker (`uv run pyright ./hexai`)
- **MyPy**: Standard type checker (`uv run mypy ./hexai`)
- Both run on every commit and in CI/CD pipelines

### Branch Naming Convention
Branch names must match the pattern: `^(ci|dependabot\/pip|docs|experiment|feat|fix|refactor|test)\/[A-Za-z0-9._-]+$`

Examples:
- `feat/yaml-pipeline-builder`
- `fix/validation-error-handling`

### Testing Approach
- Unit tests for domain logic
- Integration tests for orchestrator workflows
- Mock adapters for external services

### Error Handling
- Use custom exception hierarchies (e.g., `OrchestratorError`, `ValidationError`)
- Maintain error context and node information
- Event emission for error tracking

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

_For complete YAML syntax, see `.claude/yaml_guide.md`_

## External Dependencies

The framework integrates with external services through ports:
- **LLM Port**: Language model interactions (OpenAI, Anthropic, etc.)
- **Memory Port**: Persistent memory for agents
- **Database Port**: Data persistence and retrieval
- **Tool Router**: Function calling and tool execution

Use mock adapters in `hexai/adapters/mock/` for development and testing.

## Additional Documentation

For detailed information, see `.claude/` directory:
- `architecture_detail.md` - Deep dive into hexagonal architecture
- `type_hints_guide.md` - Comprehensive type checking guide
- `yaml_guide.md` - Complete YAML pipeline syntax