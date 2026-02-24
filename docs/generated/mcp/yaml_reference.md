# hexDAG YAML Manifest Reference

This reference is auto-generated from the pipeline JSON schema and compiler plugin definitions.

## Manifest Kinds

hexDAG YAML files use a Kubernetes-like manifest format. Three kinds are supported:

| Kind | Purpose |
|------|---------|
| `Pipeline` | Define a workflow with nodes and ports |
| `Macro` | Define a reusable node template |
| `Config` | Define runtime configuration |

Multiple kinds can appear in a single file using YAML multi-document syntax (`---` separator).

## kind: Pipeline

The primary manifest kind. Defines a DAG of nodes with ports and events.

### Structure

```yaml
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: my-pipeline
  description: Optional description
spec:
  ports:      # Adapter configurations
    llm:
      adapter: hexdag.stdlib.adapters.openai.OpenAIAdapter
      config:
        model: gpt-4
        api_key: ${OPENAI_API_KEY}
  nodes:      # Processing nodes
    - kind: llm_node
      metadata:
        name: analyzer
      spec:
        prompt_template: "Analyze: {{input}}"
      dependencies: []
  events: {}  # Optional event handlers
```

## Node Types

| Node Kind | Description |
|-----------|-------------|
| `agent_node` | Specification for agent_node type |
| `api_call_node` | Specification for api_call_node type |
| `composite_node` | Specification for composite_node type |
| `data_node` | Specification for data_node type |
| `expression_node` | Specification for expression_node type |
| `function_node` | Specification for function_node type |
| `llm_node` | Specification for llm_node type |
| `port_call_node` | Specification for port_call_node type |
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

### api_call_node

Specification for api_call_node type

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `headers` | string | No | Request headers. Values support ``{{v... |
| `json_body` | string | No | JSON body for POST/PUT/PATCH requests. |
| `method` | string | No | HTTP method: GET, POST, PUT, DELETE, ... |
| `output_schema` | string | No | Optional schema for output validation. |
| `params` | string | No | Query parameters for GET requests. |
| `port` | string | No | Port name to use (default: ``"api_cal... |
| `url` | string | No | URL or path template. Supports ``{{va... |

**Example:**

```yaml
- kind: api_call_node
  metadata:
    name: my_api_call
  spec:
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
| `output_schema` | object | string | No | Output schema for validation (if None... |

**Example:**

```yaml
- kind: function_node
  metadata:
    name: my_function
  spec:
  dependencies: []
```

### llm_node

Specification for llm_node type

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `output_schema` | string | No | Expected output schema for structured... |
| `parse_json` | string | No | If True, parse the LLM response as JS... |
| `parse_strategy` | string | No | JSON parsing strategy: "json", "json_... |
| `prompt_template` | string | No | Template for the user prompt. Support... |
| `system_prompt` | string | No | System message to prepend to the conv... |
| `template` | string | No |  |

**Example:**

```yaml
- kind: llm_node
  metadata:
    name: my_llm
  spec:
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

## kind: Macro

Define reusable node templates with parameters. Macros are expanded inline at build time.

### Structure

```yaml
apiVersion: hexdag/v1
kind: Macro
metadata:
  name: retry_workflow
  description: Retry logic with exponential backoff
parameters:
  - name: max_retries
    type: int
    default: 3
  - name: fn
    type: str
    required: true
nodes:
  - kind: function_node
    metadata:
      name: "{{name}}_attempt"
    spec:
      fn: "{{fn}}"
outputs:           # Optional output mapping
  result: "{{name}}_attempt.result"
```

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `metadata.name` | string | Yes | Unique macro name |
| `metadata.description` | string | No | Human-readable description |
| `parameters` | list | No | Parameter definitions |
| `parameters[].name` | string | Yes | Parameter name |
| `parameters[].type` | string | No | Type hint (str, int, float, bool) |
| `parameters[].default` | any | No | Default value |
| `parameters[].required` | bool | No | Whether parameter is required |
| `nodes` | list | Yes | Node definitions (same as Pipeline nodes) |
| `outputs` | object | No | Output field mappings |

### Using Macros

```yaml
# Define macro in same file or separate file
---
kind: Macro
metadata:
  name: retry_workflow
parameters:
  - name: fn
    required: true
nodes:
  - kind: function_node
    metadata:
      name: "{{name}}_run"
    spec:
      fn: "{{fn}}"
---
kind: Pipeline
metadata:
  name: my-pipeline
spec:
  nodes:
    - kind: retry_workflow     # Use macro by name
      metadata:
        name: fetch_data
      spec:
        fn: myapp.fetch
```

## kind: Config

Define runtime configuration inline alongside pipelines, or in a standalone file.

### Structure

```yaml
apiVersion: hexdag/v1
kind: Config
metadata:
  name: dev-config
spec:
  modules:
    - myapp.adapters
    - myapp.nodes
  plugins:
    - hexdag-openai
  dev_mode: true
  logging:
    level: DEBUG
    format: rich
  orchestrator:
    max_concurrent_nodes: 5
    default_node_timeout: 60.0
  limits:
    max_llm_calls: 100
    max_cost_usd: 10.0
  caps:
    deny: ["secret"]
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `modules` | list[str] | `[]` | Module paths to load |
| `plugins` | list[str] | `[]` | Plugin names to load |
| `dev_mode` | bool | `false` | Enable development mode |
| `logging.level` | str | `INFO` | Log level (DEBUG/INFO/WARNING/ERROR) |
| `logging.format` | str | `structured` | console, json, structured, dual, rich |
| `logging.output_file` | str | `null` | File path for log output |
| `logging.use_color` | bool | `true` | Use ANSI color codes |
| `logging.use_rich` | bool | `false` | Use Rich for enhanced console output |
| `logging.dual_sink` | bool | `false` | Pretty console + structured JSON |
| `orchestrator.max_concurrent_nodes` | int | — | Max parallel node execution |
| `orchestrator.default_node_timeout` | float | — | Per-node timeout in seconds |
| `limits.max_total_tokens` | int | `null` | Max tokens across all LLM calls |
| `limits.max_llm_calls` | int | `null` | Max number of LLM calls |
| `limits.max_tool_calls` | int | `null` | Max number of tool calls |
| `limits.max_cost_usd` | float | `null` | Max cost in USD |
| `limits.warning_threshold` | float | `0.8` | Fraction at which to warn |
| `caps.default_set` | list[str] | `null` | Default capability allowlist |
| `caps.deny` | list[str] | `null` | Always-denied capabilities |

### Config Loading Priority

Configuration is loaded in this order (later overrides earlier):

1. Built-in defaults
2. `pyproject.toml` `[tool.hexdag]` section
3. `kind: Config` YAML file (explicit path or `HEXDAG_CONFIG` env var)
4. Inline `kind: Config` document in multi-doc YAML
5. Environment variables (`HEXDAG_LOG_LEVEL`, `HEXDAG_DEV_MODE`, etc.)
