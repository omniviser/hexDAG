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
```

### Code Quality
```bash
# Run all pre-commit hooks
uv run pre-commit run --all-files

# Individual tools
uv run black hexai/                    # Code formatting
uv run isort hexai/                    # Import sorting
uv run ruff check hexai/ --fix         # Linting with auto-fix
uv run ruff format hexai/              # Ruff formatting
uv run mypy hexai/                     # Type checking
uv run bandit -r hexai                 # Security scanning
uv run deptry .                        # Dependency analysis
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

## Development Guidelines

### Code Standards
- **Python 3.12+** required
- **Type hints** mandatory for all public APIs
- **Pydantic models** for all data structures
- **Async-first** design - all operations should be async
- **Event-driven** - emit events for observability
- **Hexagonal architecture** - maintain clean boundaries

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

## External Dependencies

The framework integrates with external services through ports:
- **LLM Port**: Language model interactions (OpenAI, Anthropic, etc.)
- **Memory Port**: Persistent memory for agents
- **Database Port**: Data persistence and retrieval
- **Tool Router**: Function calling and tool execution

Use mock adapters in `hexai/adapters/mock/` for development and testing.
