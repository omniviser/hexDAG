# Testing with Mock Adapters - No API Keys Required

The hexDAG MCP server fully supports **mock adapters** for development and testing without real API keys. This enables:

- ✅ **Local development** without API costs
- ✅ **CI/CD testing** without exposing secrets
- ✅ **Behavior prototyping** before production
- ✅ **Unit testing** agent logic

## ✅ Verification Results

### Test 1: MCP Builds Pipelines with Mock Adapters ✅

The MCP server's `build_yaml_pipeline_interactive` successfully generates YAML with mock adapters:

```yaml
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-mock-pipeline
spec:
  ports:
    llm:
      adapter: plugin:mock_llm  # ✅ Mock LLM adapter
      config:
        responses:
          - "Mock response 1"
          - "Mock response 2"
    tool_router:
      adapter: plugin:mock_tool_router  # ✅ Mock tool router
      config:
        available_tools: [search, calculate]
  nodes:
    - kind: macro_invocation
      metadata:
        name: agent
      spec:
        macro: core:reasoning_agent
        config:
          main_prompt: "Research: {{query}}"
          max_steps: 2
          allowed_tools: [search, calculate]
      dependencies: []
```

**Status:** ✅ **PASS** - MCP generates valid YAML with mock adapters

### Test 2: Environment Variable Handling ✅

The MCP server correctly preserves environment variables in generated YAML:

```yaml
ports:
  llm:
    adapter: core:openai
    config:
      api_key: ${OPENAI_API_KEY}  # ✅ Preserved as template
      model: ${MODEL_NAME}         # ✅ Not resolved at build time
```

**Status:** ✅ **PASS** - Environment variables preserved for runtime resolution

### Test 3: Mock Adapters Available in Registry ✅

Mock adapters are registered and discoverable via MCP tools:

```python
# Via MCP
list_adapters(port_type="llm")
# Returns: plugin:mock_llm

list_adapters(port_type="tool_router")
# Returns: plugin:mock_tool_router
```

**Available Mock Adapters:**
- ✅ `plugin:mock_llm` - Mock LLM with configurable responses
- ✅ `plugin:mock_tool_router` - Mock tool execution
- ✅ `plugin:mock_database` - Mock database operations
- ✅ `plugin:mock_embedding` - Mock embedding generation

**Status:** ✅ **PASS** - All mock adapters discoverable

## 📋 Mock Adapter Capabilities

### MockLLM Adapter

**Configuration:**
```yaml
llm:
  adapter: plugin:mock_llm
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

**Use Cases:**
- Testing multi-turn conversations
- Simulating different LLM behaviors
- Performance testing without API calls

### MockToolRouter Adapter

**Configuration:**
```yaml
tool_router:
  adapter: plugin:mock_tool_router
  config:
    available_tools: [search, calculate, get_weather]
    delay_seconds: 0.05  # Optional: simulate network delay
```

**Built-in Mock Tools:**
- `search` - Returns mock search results
- `calculate` - Evaluates math expressions (safe eval)
- `get_weather` - Returns mock weather data

**Features:**
- Realistic response structures
- Call history tracking
- Configurable tool availability
- Error simulation for unknown tools

### MockDatabase Adapter

**Configuration:**
```yaml
database:
  adapter: plugin:mock_database
  config:
    initial_data:
      users: [{id: 1, name: "Alice"}]
```

**Features:**
- In-memory key-value store
- Query simulation
- Transaction tracking

## 🚀 Usage Examples

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
      adapter: plugin:mock_llm
      config:
        responses:
          - "Let me search for that. INVOKE_TOOL: search(query='Python asyncio')"
          - "Based on the results, asyncio is Python's library for asynchronous I/O."
    tool_router:
      adapter: plugin:mock_tool_router
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

**Run without API keys:**
```bash
uv run python -c "
from hexdag.core.pipeline_builder.yaml_builder import YamlPipelineBuilder
builder = YamlPipelineBuilder()
graph, config = builder.build_from_yaml_file('mock-qa-agent.yaml')
print(f'Built pipeline with {len(graph.nodes)} nodes')
"
```

### Example 2: CI/CD Testing

**GitHub Actions Workflow:**
```yaml
name: Test Agents
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install dependencies
        run: uv sync

      - name: Test agents with mocks (no API keys)
        run: |
          # Uses mock adapters - no secrets needed
          uv run pytest tests/agents/ -v
        env:
          USE_MOCK_ADAPTERS: "true"
```

### Example 3: Development Workflow

**Local Development (No API Keys):**
```python
# examples/dev_with_mocks.py
import asyncio
from hexdag.builtin.adapters.mock import MockLLM, MockToolRouter
from hexdag.core.orchestration.orchestrator import Orchestrator
from hexdag.core.pipeline_builder.yaml_builder import YamlPipelineBuilder

async def develop_agent():
    # Load your agent YAML
    builder = YamlPipelineBuilder()
    graph, _ = builder.build_from_yaml_file("my_agent.yaml")

    # Use mocks for development
    mock_llm = MockLLM(responses=[
        "Analyzing the input...",
        "Here's my conclusion based on the data."
    ])

    mock_tools = MockToolRouter(available_tools=["search", "analyze"])

    # Run without real APIs
    orchestrator = Orchestrator()
    results = await orchestrator.aexecute(
        graph,
        initial_input={"query": "test"},
        ports={"llm": mock_llm, "tool_router": mock_tools}
    )

    print(results)

# Test your agent logic without API costs
asyncio.run(develop_agent())
```

**Switch to Production:**
```python
# Same code, just swap adapters
from hexdag.builtin.adapters.openai import OpenAIAdapter

# Production
real_llm = OpenAIAdapter(api_key=os.environ["OPENAI_API_KEY"])
results = await orchestrator.aexecute(graph, initial_input=input, ports={"llm": real_llm})
```

## 🎯 Development Best Practices

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

### 3. Environment-Based Adapter Selection

```python
# config.py
import os

def get_llm_adapter():
    if os.environ.get("USE_MOCK_ADAPTERS"):
        from hexdag.builtin.adapters.mock import MockLLM
        return MockLLM(responses=["Mock response"])
    else:
        from hexdag.builtin.adapters.openai import OpenAIAdapter
        return OpenAIAdapter(api_key=os.environ["OPENAI_API_KEY"])
```

## 🔍 Verification Commands

```bash
# Test 1: Verify mock adapters can be imported
uv run python -c "
from hexdag.builtin.adapters.mock import MockLLM, MockToolRouter
print('Mock LLM:', MockLLM)
print('Mock Tool Router:', MockToolRouter)
"

# Test 2: Build pipeline with mocks
uv run python examples/mcp/test_mcp_with_mocks.py

# Test 3: List all mock adapters via MCP
uv run python -c "
from hexdag.mcp_server import list_adapters
import json
adapters = json.loads(list_adapters())
for port, adapters_list in adapters.items():
    mock_adapters = [a for a in adapters_list if 'mock' in a['name']]
    if mock_adapters:
        print(f'{port}: {[a[\"name\"] for a in mock_adapters]}')
"
```

## 📊 Test Results Summary

| Test | Status | Details |
|------|--------|---------|
| MCP builds with mock adapters | ✅ PASS | Generates valid YAML with `plugin:mock_*` |
| Environment variable preservation | ✅ PASS | `${VAR}` templates preserved correctly |
| Mock adapter registry | ✅ PASS | All 4 mock adapters discoverable |
| Mock response configuration | ✅ PASS | Sequential responses work correctly |
| Tool call simulation | ✅ PASS | Mock tools return realistic data |

## 🎉 Conclusion

The hexDAG MCP server **fully supports mock adapters** for API-free development and testing:

✅ **Build Phase**: MCP generates YAML with mock adapter references
✅ **Validation Phase**: YAML validates successfully with mock adapters
✅ **Registry Phase**: Mock adapters are discoverable via MCP tools
✅ **Configuration Phase**: Mock behavior is configurable via YAML

**Recommendation:** Use mock adapters for:
- Local development (save API costs)
- Automated testing (no secrets in CI/CD)
- Rapid prototyping (instant feedback)
- Learning hexDAG (no API key barriers)

Switch to real adapters only when ready for production!

---

**Test Suite:** [test_mcp_with_mocks.py](test_mcp_with_mocks.py)
**Mock Implementations:** [hexdag/builtin/adapters/mock/](../../hexdag/builtin/adapters/mock/)
