# hexDAG -- Developer-First Workflow Engine for AI Agents

[![PyPI version](https://img.shields.io/pypi/v/hexdag.svg)](https://pypi.org/project/hexdag/)
[![Python 3.12](https://img.shields.io/badge/python-3.12.*-blue.svg)](https://www.python.org/downloads/)
[![uv: Python package manager](https://img.shields.io/badge/uv-fastest--python--installer-blueviolet?logo=python&logoColor=white)](https://github.com/astral-sh/uv)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)

> Compose n8n-like automations in YAML or Python, run LangGraph-style agent flows
> as typed DAGs, and ship them with observability, replay, and human approval.

## Why hexDAG?

- **Visual/workflow mindset like n8n** -- define workflows in YAML, version them in git
- **Programmable control like LangGraph** -- typed DAGs with conditional branching, loops, and agent reasoning
- **Runtime guarantees like production infrastructure** -- observability, replay, human approval gates
- **YAML or Python source of truth** -- declarative-first, code when you need it
- **Typed node contracts** -- Pydantic validation at every boundary
- **Event stream, replay, audit** -- every execution step tracked

hexDAG is async-first, type-safe with Pydantic, and built on hexagonal architecture
so you can swap any infrastructure dependency without touching business logic.

---

## How hexDAG Compares

| Tool | Best for | Limitation hexDAG targets |
|---|---|---|
| n8n | Visual automations | Harder to version, test, type, and embed deeply in Python apps |
| LangGraph | Agent control flow | Less workflow/product-infra oriented out of the box |
| Prefect | Data/infra workflows | Not AI-agent native |
| Airflow | Scheduled batch DAGs | Heavy, not agent-native, less interactive |
| **hexDAG** | **Programmable AI workflows** | **Code/YAML-first, typed, observable, replayable** |

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
  kernel/   -- Core primitives. Protocols, domain models, orchestration.
  stdlib/   -- Standard library. Built-in nodes, adapters, macros, libs.
  drivers/  -- Low-level infrastructure. Executor, observer, spawner.
  api/      -- Public API. MCP tools + Studio REST endpoints.
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

| Entity   | Kernel Contract                | Stdlib Builtins                               | User Custom        |
| -------- | ------------------------------ | --------------------------------------------- | ------------------ |
| Ports    | `kernel/ports/llm.py`          | --                                            | `myapp.ports.X`    |
| Adapters | Protocol in `kernel/ports/`    | `stdlib/adapters/openai/`                     | `myapp.adapters.X` |
| Nodes    | `NodeSpec` + `BaseNodeFactory` | `LLMNode`, `AgentNode`, `FunctionNode`        | `myapp.nodes.X`    |
| Macros   | (convention)                   | `ReasoningAgent`, `ConversationAgent`         | `myapp.macros.X`   |
| Prompts  | (convention)                   | `tool_prompts`, `error_correction`            | `myapp.prompts.X`  |
| Libs     | `HexDAGLib` base class         | `ProcessRegistry`, `EntityState`, `Scheduler` | `myapp.lib.X`      |

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

| Lib                 | Purpose                                                                                      |
| ------------------- | -------------------------------------------------------------------------------------------- |
| **ProcessRegistry** | Track pipeline runs -- status, duration, results, parent/child relationships                 |
| **EntityState**     | Declarative state machines for business entities with validated transitions and audit trails |
| **Scheduler**       | Delayed and recurring pipeline execution via asyncio timers                                  |
| **DatabaseTools**   | Agent-callable SQL query tools wrapping any `SupportsQuery` adapter                          |

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
Built-in macros: `ReasoningAgent`, `ConversationAgent`, `LLMMacro`.

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

| Command                    | Purpose                                   |
| -------------------------- | ----------------------------------------- |
| `hexdag init`              | Initialize a new hexDAG project           |
| `hexdag pipeline validate` | Validate a YAML pipeline                  |
| `hexdag pipeline execute`  | Execute a pipeline                        |
| `hexdag lint`              | Lint YAML for best practices and security |
| `hexdag create`            | Create pipeline templates from schemas    |
| `hexdag build`             | Build Docker containers for pipelines     |
| `hexdag studio`            | Launch the visual pipeline editor         |
| `hexdag plugins`           | Manage plugins and adapters               |
| `hexdag docs`              | Generate and serve documentation          |

---

## Documentation & Learning

### Interactive Notebooks

| Notebook                                                                        | Topic                        | Time   |
| ------------------------------------------------------------------------------- | ---------------------------- | ------ |
| [01. Introduction](notebooks/01_introduction.ipynb)                             | Your first pipeline          | 15 min |
| [02. YAML Pipelines](notebooks/02_yaml_pipelines.ipynb)                         | Declarative workflows        | 25 min |
| [03. Practical Workflow](notebooks/03_practical_workflow.ipynb)                 | Real-world patterns          | 30 min |
| [06. Dynamic Reasoning Agent](notebooks/06_dynamic_reasoning_agent.ipynb)       | Advanced agent patterns      | --     |
| [YAML Includes & Composition](notebooks/03_yaml_includes_and_composition.ipynb) | Modular pipeline composition | --     |

### Docs

- [Architecture](docs/ARCHITECTURE.md) -- System architecture and design philosophy
- [Roadmap](docs/ROADMAP.md) -- Development roadmap and planned kernel extensions
- [Quick Start](docs/getting-started/quickstart.md) -- Build your first workflow
- [All Documentation](docs/README.md) -- Full documentation index

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

## Design Principles

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
