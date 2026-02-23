# hexDAG -- Operating System for AI Agents

[![PyPI version](https://img.shields.io/pypi/v/hexdag.svg)](https://pypi.org/project/hexdag/)
[![Python 3.12](https://img.shields.io/badge/python-3.12.*-blue.svg)](https://www.python.org/downloads/)
[![uv: Python package manager](https://img.shields.io/badge/uv-fastest--python--installer-blueviolet?logo=python&logoColor=white)](https://github.com/astral-sh/uv)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)

> Just as Linux provides processes, syscalls, drivers, and `/lib` so programs don't
> reinvent the wheel, hexDAG provides **pipelines**, **ports**, **drivers**, and a
> **standard library** so AI agents don't reinvent orchestration.

hexDAG transforms complex AI workflows into deterministic, testable, and maintainable
systems through declarative YAML pipelines and DAG-based execution. It is async-first,
type-safe with Pydantic, and built on hexagonal architecture so you can swap any
infrastructure dependency without touching business logic.

---

## The OS Analogy

| Linux | hexDAG | Purpose |
|-------|--------|---------|
| Kernel | `kernel/` | Core execution engine, system call interfaces (Protocols), domain models |
| System calls | `kernel/ports/` | Contracts for external capabilities (LLM, Memory, Database, etc.) |
| Drivers | `drivers/` | Low-level infrastructure (executor, observer manager, pipeline spawner) |
| `/lib` | `stdlib/` | Standard library -- built-in nodes, adapters, macros, system libs |
| Processes | Pipeline runs | Tracked by `ProcessRegistry` (like `ps`) |
| `fork`/`exec` | `PipelineSpawner` | Launch sub-pipelines from within a running pipeline |
| Process scheduler | `Scheduler` | Delayed and recurring pipeline execution |
| State machines | `EntityState` | Business entity lifecycle management |
| `/usr/bin` | `api/` | User-facing tools (MCP + Studio REST) |
| Shell | `cli/` | Command-line interface (`hexdag init`, `hexdag lint`, etc.) |

---

## Quick Start

### Installation

```bash
# Install from PyPI
pip install hexdag

# Or with uv (recommended)
uv pip install hexdag

# With optional dependencies
pip install hexdag[openai]      # OpenAI adapter
pip install hexdag[anthropic]   # Anthropic adapter
pip install hexdag[all]         # Everything
```

#### Development

```bash
git clone https://github.com/omniviser/hexdag.git
cd hexdag
uv sync
```

### Your First Pipeline

Define a workflow in YAML:

```yaml
# research_agent.yaml
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: research-workflow
  description: AI-powered research assistant
spec:
  ports:
    llm:
      adapter: hexdag.stdlib.adapters.openai.OpenAIAdapter
      config:
        model: gpt-4

  nodes:
    - kind: agent_node
      metadata:
        name: researcher
      spec:
        initial_prompt_template: "Research the topic: {{topic}}"
        max_steps: 5
        available_tools: ["web_search", "summarize"]
      dependencies: []

    - kind: llm_node
      metadata:
        name: analyst
      spec:
        prompt_template: |
          Analyze the research findings: {{researcher.results}}
          Provide actionable insights.
      dependencies: [researcher]

    - kind: function_node
      metadata:
        name: formatter
      spec:
        fn: "myapp.reports.format_report"
        input_mapping:
          findings: "researcher.results"
          insights: "analyst.output"
      dependencies: [researcher, analyst]
```

Run it:

```python
from hexdag import PipelineRunner

runner = PipelineRunner()
result = await runner.run("research_agent.yaml", input_data={"topic": "AI trends 2025"})
```

With port overrides for testing:

```python
from hexdag import PipelineRunner, MockLLM

runner = PipelineRunner(port_overrides={"llm": MockLLM(responses="Mock analysis result")})
result = await runner.run("research_agent.yaml", input_data={"topic": "AI trends 2025"})
```

Or validate without executing:

```python
runner = PipelineRunner()
issues = await runner.validate(pipeline_path="research_agent.yaml")
# [] means valid
```

---

## Architecture

```
hexdag/
  kernel/   -- Core primitives. Protocols, domain models, orchestration.     (/kernel)
  stdlib/   -- Standard library. Built-in nodes, adapters, macros, libs.     (/lib)
  drivers/  -- Low-level infrastructure. Executor, observer, spawner.        (/drivers)
  api/      -- Public API. MCP tools + Studio REST endpoints.                (/usr/bin)
  cli/      -- CLI commands (hexdag init, hexdag lint, hexdag studio, etc.)
```

The **kernel** defines contracts and depends on nothing external. The **stdlib** ships
implementations users interact with directly. **Drivers** provide low-level infrastructure
the orchestrator needs internally. The **API** exposes everything to users via MCP and REST.

### Decision Rules

When adding something new:

1. Defines a Protocol or domain model? --> `kernel/`
2. Implements a Protocol for users to use or extend? --> `stdlib/`
3. Low-level infrastructure the orchestrator needs internally? --> `drivers/`
4. User-facing function (MCP tool or REST endpoint)? --> `api/`

---

## Uniform Entity Pattern

All framework entities follow one pattern: **kernel defines contract, stdlib ships builtins, users write their own.**

| Entity | Kernel Contract | Stdlib Builtins | User Custom |
|--------|----------------|-----------------|-------------|
| Ports | `kernel/ports/llm.py` | -- | `myapp.ports.X` |
| Adapters | Protocol in `kernel/ports/` | `stdlib/adapters/openai/` | `myapp.adapters.X` |
| Nodes | `NodeSpec` + `BaseNodeFactory` | `LLMNode`, `AgentNode`, `FunctionNode` | `myapp.nodes.X` |
| Macros | (convention) | `ReasoningAgent`, `ConversationAgent` | `myapp.macros.X` |
| Prompts | (convention) | `tool_prompts`, `error_correction` | `myapp.prompts.X` |
| Libs | `HexDAGLib` base class | `ProcessRegistry`, `EntityState`, `Scheduler` | `myapp.lib.X` |

Components are referenced by full Python module path in YAML:

```yaml
nodes:
  - kind: hexdag.stdlib.nodes.LLMNode      # Full path
  - kind: llm_node                          # Built-in alias
  - kind: myapp.nodes.CustomNode            # User custom
```

---

## System Libraries (Libs)

Libs are system-level capabilities for multi-pipeline coordination. Every public async
method on a `HexDAGLib` subclass auto-becomes an agent-callable tool.

| Lib | Linux Analogy | Purpose |
|-----|--------------|---------|
| **ProcessRegistry** | `ps` | Track pipeline runs -- status, duration, results, parent/child relationships |
| **EntityState** | State machines | Declarative state machines for business entities with validated transitions and audit trails |
| **Scheduler** | `cron` / `at` | Delayed and recurring pipeline execution via asyncio timers |
| **DatabaseTools** | `sqlite3` CLI | Agent-callable SQL query tools wrapping any `SupportsQuery` adapter |

```python
from hexdag.kernel.lib_base import HexDAGLib
from hexdag.kernel.ports.data_store import SupportsKeyValue

class OrderManager(HexDAGLib):
    def __init__(self, store: SupportsKeyValue) -> None:
        self._store = store

    async def acreate_order(self, customer_id: str, items: list[dict]) -> str:
        """Create a new order. Auto-exposed as agent tool."""
        ...

    async def aget_order(self, order_id: str) -> dict:
        """Get order by ID. Auto-exposed as agent tool."""
        ...
```

---

## Key Features

### YAML-First Pipelines
Declarative workflows versioned in git. Environment-specific configs for dev/staging/prod.
Jinja2 templating, `!include` tags for modular composition, and automatic field mapping
between nodes.

### Pipeline Linting
Static analysis for YAML pipelines: cycle detection, hardcoded secret scanning,
missing timeout/retry warnings, naming conventions, and unused output detection.

```bash
hexdag lint my_pipeline.yaml
```

### Macro System
Reusable pipeline templates that expand into full DAG subgraphs at build time.
Built-in macros: `ReasoningAgent`, `ConversationAgent`, `LLMMacro`, `ToolMacro`.

### Event-Driven Observability
Comprehensive event system (`PipelineStarted`, `NodeCompleted`, `NodeFailed`, etc.)
with pluggable observers for logging, cost profiling, and custom monitoring.

### Hexagonal Architecture (Ports & Adapters)
Swap OpenAI for Anthropic, PostgreSQL for SQLite, or any adapter -- with one config
line. Business logic stays pure. Test everything with mock adapters.

### Pydantic Validation Everywhere
Type safety at every layer. Automatic schema compatibility checking between connected
nodes. Runtime type coercion and validation.

---

## MCP Server

hexDAG includes a built-in [Model Context Protocol](https://modelcontextprotocol.io)
server that exposes pipeline building capabilities to Claude Code, Cursor, and other
LLM-powered editors:

```bash
# Install MCP dependencies
uv sync --extra mcp

# Configure in Claude Desktop
```

```json
{
  "mcpServers": {
    "hexdag": {
      "command": "uv",
      "args": ["run", "python", "-m", "hexdag", "--mcp"]
    }
  }
}
```

The MCP server provides tools to:
- List available nodes, adapters, tools, and macros from your registry
- Build and validate YAML pipelines interactively
- Get component schemas and documentation
- Manage processes (spawn, schedule, track pipeline runs)
- Auto-discover custom plugins from `HEXDAG_PLUGIN_PATHS`

### Custom Plugin Discovery

hexDAG supports three levels of component discovery:
1. **Builtin** -- Core adapters and nodes from `hexdag.stdlib`
2. **Plugins** -- Community plugins from the `hexdag_plugins` namespace
3. **User-authored** -- Your custom components via `HEXDAG_PLUGIN_PATHS`

```bash
export HEXDAG_PLUGIN_PATHS="./my_adapters:./my_nodes"
```

---

## CLI

| Command | Purpose |
|---------|---------|
| `hexdag init` | Initialize a new hexDAG project |
| `hexdag pipeline validate` | Validate a YAML pipeline |
| `hexdag pipeline execute` | Execute a pipeline |
| `hexdag lint` | Lint YAML for best practices and security |
| `hexdag create` | Create pipeline templates from schemas |
| `hexdag build` | Build Docker containers for pipelines |
| `hexdag studio` | Launch the visual pipeline editor |
| `hexdag plugins` | Manage plugins and adapters |
| `hexdag docs` | Generate and serve documentation |

---

## Documentation & Learning

### Interactive Notebooks

| Notebook | Topic | Time |
|----------|-------|------|
| [01. Introduction](notebooks/01_introduction.ipynb) | Your first pipeline | 15 min |
| [02. YAML Pipelines](notebooks/02_yaml_pipelines.ipynb) | Declarative workflows | 25 min |
| [03. Practical Workflow](notebooks/03_practical_workflow.ipynb) | Real-world patterns | 30 min |
| [06. Dynamic Reasoning Agent](notebooks/06_dynamic_reasoning_agent.ipynb) | Advanced agent patterns | -- |
| [YAML Includes & Composition](notebooks/03_yaml_includes_and_composition.ipynb) | Modular pipeline composition | -- |

### Docs

- [Architecture](docs/ARCHITECTURE.md) -- System architecture and the four layers
- [Philosophy & Design](docs/PHILOSOPHY.md) -- Six pillars and design principles
- [Hexagonal Architecture](docs/HEXAGONAL_ARCHITECTURE.md) -- Ports and adapters explained
- [Implementation Guide](docs/IMPLEMENTATION_GUIDE.md) -- Production-ready workflows
- [Plugin System](docs/PLUGIN_SYSTEM.md) -- Custom component development
- [CLI Reference](docs/CLI_REFERENCE.md) -- Complete CLI documentation
- [Roadmap](docs/ROADMAP.md) -- Development roadmap and planned kernel extensions

### Examples

- [MCP Research Agent](examples/mcp/) -- Deep research agent with environment configs
- [Demo: Startup Pitch](examples/demo/) -- Live demo with rich terminal UI

---

## Development

```bash
# Setup
uv sync
uv run pre-commit install

# Test
uv run pytest
uv run pytest --cov=hexdag --cov-report=term-missing

# Code quality
uv run ruff check hexdag/ --fix
uv run pyright ./hexdag
uv run pre-commit run --all-files
```

---

## The Six Pillars

1. **Async-First Architecture** -- Non-blocking execution for maximum performance
2. **Event-Driven Observability** -- Real-time monitoring via comprehensive event system
3. **Pydantic Validation Everywhere** -- Type safety at every layer
4. **Hexagonal Architecture** -- Clean separation of business logic and infrastructure
5. **Composable Declarative Files** -- Complex workflows from simple YAML components
6. **DAG-Based Orchestration** -- Intelligent dependency management and parallelization

---

## License

Apache License 2.0 -- see [LICENSE](LICENSE) for details.

---

Built for the AI community by the hexDAG team.
