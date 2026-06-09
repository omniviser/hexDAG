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

## How to Build with hexDAG

### What to build

| You need... | Use |
|---|---|
| One workflow (e.g., "analyze this document") | `kind: Pipeline` — a DAG of nodes |
| Multiple workflows sharing state (e.g., "order lifecycle") | `kind: System` — bundles pipelines + state machines + shared ports |
| Entities that change state (e.g., orders, tickets) | `spec.state_machines` on Pipeline or System |

### Which node to use

| You want to... | Node | Example |
|---|---|---|
| Ask an LLM a question | `llm_node` | Summarize, classify, extract |
| Let an LLM reason + use tools | `agent_node` | Research, multi-step decisions |
| Run a Python function | `function_node` | Transform data, call APIs |
| Compute values from upstream data | `expression_node` | `"order.total * 1.1"` |
| Call a service method | `service_call_node` | `service: orders, method: save` |
| Branch or loop | `composite_node` | if/else, switch, while, for-each |
| Transition an entity state | `transition` | Move order from PENDING to SHIPPED |
| Wait for external input | `wait_node` | Human approval, webhook callback |
| Make an HTTP request | `api_call_node` | REST API calls |

### What you wire up

| Concept | What it is | When you need it |
|---|---|---|
| **Ports** | Abstract contracts (LLM, DataStore, etc.) | Always — pipelines declare which ports they need |
| **Adapters** | Concrete implementations (OpenAI, SQLite, Mock) | Always — you pick one adapter per port |
| **Middleware** | Transparent wrappers (retry, timeout, cache) | When you need resilience |
| **Services** | Your business logic with `@tool`/`@step` | When agents need to call your code |

### How data flows

Upstream node outputs are automatically available downstream. No manual wiring needed.

```yaml
nodes:
  - kind: llm_node
    metadata: { name: analyzer }
    spec:
      human_message: "Analyze: {{ $input.topic }}"

  - kind: expression_node
    metadata: { name: compute }
    spec:
      expressions:
        score: "analyzer.confidence * 100"    # auto-detected dependency
```

### Control flow

`composite_node` supports `if-else`, `switch`, `while`, `for-each`, and `times`:

```yaml
- kind: composite_node
  metadata: { name: route }
  spec:
    mode: if-else
    condition: "$input.priority == 'urgent'"
    body: "myapp.handle_urgent"
    else_body: "myapp.handle_normal"
```

See the [Framework Guide](docs/GUIDE.md) for all composite modes, YAML syntax, services, macros, and entity lifecycle.

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

For the complete list of public symbols, version policy, and import paths, see [PUBLIC_API.md](docs/PUBLIC_API.md).

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
| Services | `Service` + `@tool`/`@step`    | `ProcessRegistry`, `EntityState`, `PipelineMemory` | `myapp.services.X` |

Components are referenced by full Python module path in YAML:

```yaml
nodes:
  - kind: hexdag.stdlib.nodes.LLMNode      # Full path
  - kind: llm_node                          # Built-in alias
  - kind: myapp.nodes.CustomNode            # User custom
```

---

## Services

Services wrap port-backed business logic behind `@tool` and `@step` decorators.
Agents call `@tool` methods during ReAct reasoning; `@step` methods run as deterministic DAG nodes via `service_call_node`.

```yaml
spec:
  services:
    orders:
      class: myapp.services.OrderService
      config:
        store: { ref: main_store }

  nodes:
    # @step methods — deterministic DAG nodes
    - kind: service_call_node
      metadata: { name: save }
      spec:
        service: orders
        method: save_order

    # @tool methods — available to agents via service_name:method_name
    - kind: agent_node
      metadata: { name: order_agent }
      spec:
        initial_prompt_template: "Handle order {{ $input.order_id }}"
        available_tools: ["orders:get_order", "orders:validate_order"]
        max_steps: 5
```

The Python side:

```python
from hexdag.kernel.service import Service, tool, step
from hexdag.kernel.ports.data_store import SupportsKeyValue

class OrderService(Service):
    def __init__(self, store: SupportsKeyValue) -> None:
        self._store = store

    @tool
    async def get_order(self, order_id: str) -> dict:
        """Agent-callable during ReAct reasoning."""
        ...

    @step
    async def save_order(self, order_id: str, data: dict) -> dict:
        """Deterministic DAG node via service_call_node."""
        ...
```

| Built-in Service    | Purpose                                                                                     |
| ------------------- | ------------------------------------------------------------------------------------------- |
| **ProcessRegistry** | Track pipeline runs -- status, duration, results, parent/child relationships                |
| **EntityState**     | Declarative state machines for business entities with validated transitions and audit trails |
| **PipelineMemory**  | Run-scoped key-value store (auto-registered for every pipeline run)                         |

See [PUBLIC_API.md > Services](docs/PUBLIC_API.md#services--extension-points) for all service-related symbols.

---

## Middleware

Transparent wrappers that add resilience to any port. Declared in YAML -- nodes don't know middleware exists.

```yaml
spec:
  ports:
    llm:
      adapter: openai
      middleware:
        - hexdag.stdlib.middleware.RetryWithBackoff:
            max_retries: 3
        - hexdag.stdlib.middleware.Timeout:
            timeout_seconds: 30
        - hexdag.stdlib.middleware.ResponseCache
```

Or stack programmatically with `compose()`:

```python
from hexdag.stdlib.middleware import compose, ResponseCache, RetryWithBackoff, Timeout
from hexdag import MockLLM

llm = compose(MockLLM(), ResponseCache, RetryWithBackoff, Timeout)
```

| Middleware | Purpose |
|---|---|
| `RetryWithBackoff` | Exponential backoff retry for transient failures |
| `RateLimiter` | Token-bucket rate limiting |
| `CircuitBreaker` | Fail-fast after repeated errors (open/half-open/closed) |
| `ResponseCache` | LRU cache for identical calls |
| `Timeout` | Wall-clock timeout enforcement |
| `RoundRobin` | Load balance across multiple adapters |
| `BatchGeneration` | Batch generation control |
| `DistributedCache` | External store cache backend |

See [PUBLIC_API.md > Middleware](docs/PUBLIC_API.md#middleware-from-hexdagstdlibmiddleware-import-) for full details.

---

## Key Features

### YAML-First Pipelines
Declarative workflows versioned in git. Environment-specific configs via `!include` for
dev/staging/prod. Jinja2 templating, auto-detected dependencies between nodes, and
inline expressions in `input_mapping`. See [Framework Guide > YAML Syntax](docs/GUIDE.md#3-yaml-syntax).

### Pipeline Linting
Static analysis for YAML pipelines: cycle detection, hardcoded secret scanning,
missing timeout/retry warnings, naming conventions, and unused output detection.

```bash
hexdag lint my_pipeline.yaml
```

### Macro System
Reusable pipeline templates that expand into full DAG subgraphs at build time.
Built-in: `core:reasoning_agent`, `core:conversation_agent`, `core:llm_macro`.

```yaml
- kind: macro_invocation
  metadata: { name: research_agent }
  spec:
    macro: core:reasoning_agent
    config:
      main_prompt: "Research: {{research_question}}"
      max_steps: 10
      allowed_tools: ["research:tavily_search"]
```

See [Framework Guide > Macros](docs/GUIDE.md#7-macros-vs-nodes).

### Event-Driven Observability
Comprehensive event system (`PipelineStarted`, `NodeCompleted`, `NodeFailed`, `WaveCompleted`, etc.)
with pluggable observers for logging, cost profiling, and custom monitoring.
`ResourceAccountingObserver` tracks token usage and enforces budget limits.
See [PUBLIC_API.md > Events](docs/PUBLIC_API.md#events).

### Hexagonal Architecture (Ports & Adapters)
Swap OpenAI for Anthropic, PostgreSQL for SQLite, or any adapter -- with one config
line. Business logic stays pure. Test everything with mock adapters.
See [PUBLIC_API.md > Port Capabilities](docs/PUBLIC_API.md#port-capabilities-supports) for all port protocols and sub-protocols.

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

Key MCP tools:

| Tool | Purpose |
|---|---|
| `list_nodes` | List all node types with schemas |
| `list_adapters` | List adapters, filter by port type |
| `validate_yaml_pipeline` | Validate YAML for errors |
| `get_component_schema` | Detailed schema for a component |
| `build_yaml_pipeline_interactive` | Build pipeline from structured input |
| `explain_yaml_structure` | YAML structure documentation |

Full list: [PUBLIC_API.md > MCP Tools](docs/PUBLIC_API.md#mcp-server-tools).

Auto-discovers custom plugins from `HEXDAG_PLUGIN_PATHS`.

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
| `hexdag explain`           | Explain nodes, adapters, middleware, YAML syntax |
| `hexdag create`            | Create pipeline templates from schemas    |
| `hexdag build`             | Build Docker containers for pipelines     |
| `hexdag studio`            | Launch the visual pipeline editor         |
| `hexdag plugins`           | Manage plugins and adapters               |
| `hexdag docs`              | Generate and serve documentation          |

```bash
# Validate before pushing
hexdag lint my_pipeline.yaml

# Explore what a node type accepts
hexdag explain node llm_node

# Run a pipeline
hexdag pipeline execute my_pipeline.yaml --input '{"query": "hello"}'

# See all available adapters
hexdag explain adapter openai
```

Full CLI reference: [PUBLIC_API.md > CLI](docs/PUBLIC_API.md#cli-commands).

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

- [Framework Guide](docs/GUIDE.md) -- YAML syntax, services, macros, composite nodes, entity lifecycle
- [Public API Reference](docs/PUBLIC_API.md) -- All public symbols, version policy, import paths
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

## License

Apache License 2.0 -- see [LICENSE](LICENSE) for details.

---

Built for the AI community by the hexDAG team.
