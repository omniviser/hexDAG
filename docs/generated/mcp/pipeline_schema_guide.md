# hexDAG Pipeline Schema Reference

This reference is auto-generated from the pipeline JSON schema.

## Overview

hexDAG pipelines are defined in YAML using a Kubernetes-like structure.
The schema provides validation and IDE autocompletion support.

## Pipeline Structure

```yaml
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: my-pipeline
  description: Pipeline description
spec:
  ports: {}     # Adapter configurations
  nodes: []     # Processing nodes
  events: {}    # Event handlers
```

## Node Types

| Node Kind | Description |
|-----------|-------------|
| `agent_node` | Specification for agent_node type |
| `conditional_node` | Specification for conditional_node type |
| `function_node` | Specification for function_node type |
| `llm_node` | Specification for llm_node type |
| `loop_node` | Specification for loop_node type |
| `tool_call_node` | Specification for tool_call_node type |

### agent_node

Specification for agent_node type

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `config` | string | null | No | Agent configuration |
| `continuation_prompts` | object | null | No | Phase-specific prompts |
| `main_prompt` | string | Yes | Initial reasoning prompt |
| `output_schema` | object | string | No | Custom output schema for tool_end res... |

**Example:**

```yaml
- kind: agent_node
  metadata:
    name: my_agent
  spec:
    main_prompt: # required
  dependencies: []
```

### conditional_node

Multi-branch conditional router for workflow control flow

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `branches` | list[object] | Yes | List of condition branches evaluated ... |
| `else_action` | string | No | Default action if no branch condition... |
| `tie_break` | `"first_true`" | No | Strategy for handling multiple matchi... |

**Example:**

```yaml
- kind: conditional_node
  metadata:
    name: my_conditional
  spec:
    branches: # required
  dependencies: []
```

### function_node

Specification for function_node type

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `fn` | string | Yes | Module path string (e.g., 'myapp.proc... |

**Example:**

```yaml
- kind: function_node
  metadata:
    name: my_function
  spec:
    fn: # required
  dependencies: []
```

### llm_node

Specification for llm_node type

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `output_schema` | object | No | Expected output schema for structured... |
| `parse_json` | boolean | No | Parse the LLM response as JSON |
| `parse_strategy` | `"json`" | `"json_in_markdown`" | `"yaml`" | No | JSON parsing strategy |
| `prompt_template` | string | Yes | Template for the user prompt (Jinja2-... |
| `system_prompt` | string | No | System message to prepend to the conv... |

**Example:**

```yaml
- kind: llm_node
  metadata:
    name: my_llm
  spec:
    prompt_template: # required
  dependencies: []
```

### loop_node

Loop control node for iterative processing

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `body` | string | Yes | Module path to body function: (data, ... |
| `collect_mode` | `"last`" | `"list`" | `"reduce`" | No | How to collect results: last value, a... |
| `initial_state` | object | No | Initial state dict passed to first it... |
| `iteration_key` | string | No | Key name for current iteration number... |
| `max_iterations` | integer | No | Maximum number of iterations before s... |
| `while_condition` | string | Yes | Module path to condition function: (d... |

**Example:**

```yaml
- kind: loop_node
  metadata:
    name: my_loop
  spec:
    while_condition: # required
    body: # required
  dependencies: []
```

### tool_call_node

Specification for tool_call_node type

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `arguments` | object | null | No | Arguments to pass to the tool (defaul... |
| `tool_call_id` | string | null | No | Optional ID for tracking (from LLM to... |
| `tool_name` | string | Yes | Full module path to the tool function... |

**Example:**

```yaml
- kind: tool_call_node
  metadata:
    name: my_tool_call
  spec:
    tool_name: # required
  dependencies: []
```

## Ports Configuration

Ports connect pipelines to external services:

```yaml
spec:
  ports:
    llm:
      adapter: hexdag.builtin.adapters.openai.OpenAIAdapter
      config:
        api_key: ${OPENAI_API_KEY}
        model: gpt-4
    memory:
      adapter: hexdag.builtin.adapters.memory.InMemoryMemory
    database:
      adapter: hexdag.builtin.adapters.database.sqlite.SQLiteAdapter
      config:
        db_path: ./data.db
```

### Available Port Types

| Port | Purpose |
|------|---------|
| `llm` | Language model interactions |
| `memory` | Persistent agent memory |
| `database` | Data persistence |
| `secret` | Secret/credential management |
| `tool_router` | Tool invocation routing |

## Events Configuration

Configure event handlers for observability:

```yaml
spec:
  events:
    node_failed:
      - type: alert
        target: pagerduty
        severity: high
    pipeline_completed:
      - type: metrics
        target: datadog
```

### Event Types

| Event | When Triggered |
|-------|----------------|
| `pipeline_started` | Pipeline execution begins |
| `pipeline_completed` | Pipeline execution finishes |
| `node_started` | Node execution begins |
| `node_completed` | Node execution finishes |
| `node_failed` | Node execution fails |

### Handler Types

| Type | Purpose |
|------|---------|
| `alert` | Send alerts (PagerDuty, Slack) |
| `metrics` | Emit metrics (Datadog, Prometheus) |
| `log` | Write to logs |
| `webhook` | Call external webhooks |
| `callback` | Execute Python callbacks |

## IDE Setup

### VS Code

Add to `.vscode/settings.json`:

```json
{
  "yaml.schemas": {
    "./schemas/pipeline-schema.json": ["*.yaml", "pipelines/*.yaml"]
  }
}
```

### Schema Location

The schema file is at `schemas/pipeline-schema.json` and is auto-generated from node `_yaml_schema` attributes.
