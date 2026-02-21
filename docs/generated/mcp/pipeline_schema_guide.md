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
| `composite_node` | Specification for composite_node type |
| `data_node` | Specification for data_node type |
| `expression_node` | Specification for expression_node type |
| `function_node` | Specification for function_node type |
| `llm_node` | Specification for llm_node type |
| `port_call_node` | Specification for port_call_node type |
| `reasoning_agent_node` | Specification for reasoning_agent_node type |
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

### composite_node

Specification for composite_node type

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `body` | string | list[object] | No | Body to execute. Can be: - Module pat... |
| `body_pipeline` | string | null | No | Path to external pipeline YAML file |
| `branches` | array | null | No | List of condition branches for switch... |
| `collect` | `"list`" | `"last`" | `"first`" | No | Result collection mode (default: "list") |
| `concurrency` | integer | No | Max concurrent iterations for for-eac... |
| `condition` | string | null | No | Condition expression for while, if-el... |
| `count` | integer | null | No | Number of iterations for times mode |
| `default_node_timeout` | number | null | No |  |
| `else_action` | string | null | No | Action label for else branch (switch ... |
| `else_body` | string | list[object] | No | Body for else branch (if-else, switch... |
| `error_handling` | `"fail_fast`" | `"continue`" | `"collect`" | No | Error handling strategy (default: "fa... |
| `index_var` | string | No | Variable name for current index (defa... |
| `initial_state` | object | null | No | Initial state dict for while mode |
| `item_var` | string | No | Variable name for current item (defau... |
| `items` | string | null | No | Expression resolving to iterable for ... |
| `key_field` | string | null | No | Field to use as key for dict collection |
| `max_concurrent_nodes` | integer | No |  |
| `max_iterations` | integer | No | Safety limit for while loops (default... |
| `mode` | `"while`" | `"for-each`" | `"times`" | Yes | Control flow mode: while, for-each, t... |
| `reducer` | string | null | No | Module path to reducer function for r... |
| `state_update` | object | null | No | State update expressions for while mode |
| `strict_validation` | boolean | No |  |

**Example:**

```yaml
- kind: composite_node
  metadata:
    name: my_composite
  spec:
    mode: # required
  dependencies: []
```

### data_node

Specification for data_node type

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `output` | object | Yes | Output data to return. Values can be:... |

**Example:**

```yaml
- kind: data_node
  metadata:
    name: my_data
  spec:
    output: # required
  dependencies: []
```

### expression_node

Specification for expression_node type

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `expressions` | object | null | No | Mapping of {variable_name: expression... |
| `extract_field` | string | null | No | Field to extract from each dependency... |
| `merge_strategy` | `"dict`" | `"list`" | `"first`" | No | Strategy for merging multiple depende... |
| `output_fields` | array | null | No | Fields to include in output dict. If ... |
| `reducer` | string | string | No | Module path (e.g., "statistics.mean")... |

**Example:**

```yaml
- kind: expression_node
  metadata:
    name: my_expression
  spec:
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

### port_call_node

Specification for port_call_node type

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `fallback` | string | No | Value to return if the port is not av... |
| `has_fallback` | string | No | Set to True to enable fallback behavi... |
| `method` | string | Yes | Method name to invoke on the port |
| `output_schema` | string | No | Optional schema for validating/struct... |
| `port` | string | Yes | Name of the port to call (e.g., "data... |

**Example:**

```yaml
- kind: port_call_node
  metadata:
    name: my_port_call
  spec:
    port: # required
    method: # required
  dependencies: []
```

### reasoning_agent_node

Specification for reasoning_agent_node type

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `config` | string | null | No | Agent configuration |
| `continuation_prompts` | object | null | No | Phase-specific prompts |
| `main_prompt` | string | Yes | Initial reasoning prompt |
| `output_schema` | object | string | No | Custom output schema for tool_end res... |

**Example:**

```yaml
- kind: reasoning_agent_node
  metadata:
    name: my_reasoning_agent
  spec:
    main_prompt: # required
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
