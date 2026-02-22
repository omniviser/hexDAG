# Testing with Mock Adapters - No API Keys Required

The hexDAG MCP server fully supports **mock adapters** for development and testing without real API keys. This enables:

- Local development without API costs
- CI/CD testing without exposing secrets
- Behavior prototyping before production
- Unit testing agent logic

## Mock Adapter Capabilities

### MockLLM Adapter

**Configuration:**
```yaml
llm:
  adapter: hexdag.stdlib.adapters.mock.MockLLM
  config:
    responses:
      - "First response"
      - "Second response"
    delay_seconds: 0.1  # Optional: simulate latency
```

**Features:**
- Sequential responses (cycles through list)
- Configurable delay for latency simulation
- Call history tracking for debugging
- Error simulation support

### ToolRouter (for mock tools)

Use the base `ToolRouter` with plain functions for testing:

```python
from hexdag.kernel.ports.tool_router import ToolRouter

router = ToolRouter(tools={
    "search": lambda query="", **kw: {"results": [f"Result for: {query}"]},
    "calculate": lambda expression="", **kw: {"result": str(expression)},
})
```

In YAML, use module path strings for tool functions:

```yaml
tool_router:
  adapter: hexdag.kernel.ports.tool_router.ToolRouter
  config:
    tools: {}
```

### MockDatabase Adapter

**Configuration:**
```yaml
database:
  adapter: hexdag.stdlib.adapters.mock.MockDatabaseAdapter
  config:
    initial_data:
      users: [{id: 1, name: "Alice"}]
```

**Features:**
- In-memory key-value store
- Query simulation
- Transaction tracking

## Usage Examples

### Example 1: Simple Mock Agent

```yaml
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: mock-qa-agent
  description: Q&A agent using mocks (no API keys needed)
spec:
  ports:
    llm:
      adapter: hexdag.stdlib.adapters.mock.MockLLM
      config:
        responses:
          - "Let me search for that. INVOKE_TOOL: search(query='Python asyncio')"
          - "Based on the results, asyncio is Python's library for asynchronous I/O."
    tool_router:
      adapter: hexdag.kernel.ports.tool_router.ToolRouter
      config:
        tools: {}
  nodes:
    - kind: macro_invocation
      metadata:
        name: qa_agent
      spec:
        macro: core:reasoning_agent
        config:
          main_prompt: "Answer: {{question}}"
          max_steps: 2
          allowed_tools: [search]
      dependencies: []
```

### Example 2: Development Workflow

**Local Development (No API Keys):**
```python
from hexdag.kernel.ports.tool_router import ToolRouter
from hexdag.stdlib.adapters.mock import MockLLM
from hexdag.kernel.orchestration.orchestrator import Orchestrator
from hexdag.kernel.pipeline_builder.yaml_builder import YamlPipelineBuilder

async def develop_agent():
    builder = YamlPipelineBuilder()
    graph, _ = builder.build_from_yaml_file("my_agent.yaml")

    mock_llm = MockLLM(responses=[
        "Analyzing the input...",
        "Here's my conclusion based on the data."
    ])
    tool_router = ToolRouter(tools={
        "search": lambda query="", **kw: {"results": [f"Result for: {query}"]},
    })

    orchestrator = Orchestrator()
    results = await orchestrator.run(
        graph,
        initial_input={"query": "test"},
        additional_ports={"llm": mock_llm, "tool_router": tool_router},
    )
    print(results)
```

**Switch to Production:**
```python
from hexdag.stdlib.adapters.openai import OpenAIAdapter

real_llm = OpenAIAdapter(api_key=os.environ["OPENAI_API_KEY"])
```

## Development Best Practices

### 1. Develop with Mocks First

```bash
# Phase 1: Design and test logic with mocks
uv run python examples/my_agent_mock.py

# Phase 2: Validate with real APIs (limited runs)
export OPENAI_API_KEY="sk-..."
uv run python examples/my_agent_real.py
```

### 2. CI/CD with Mocks, Production with Real APIs

```yaml
# .github/workflows/test.yml
- name: Unit Tests (Mocks)
  run: pytest tests/unit/  # Uses mocks

- name: Integration Tests (Real APIs)
  run: pytest tests/integration/  # Uses real APIs
  if: github.ref == 'refs/heads/main'
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

---

**Test Suite:** [test_mcp_with_mocks.py](test_mcp_with_mocks.py)
**Mock Implementations:** [hexdag/stdlib/adapters/mock/](../../hexdag/stdlib/adapters/mock/)
