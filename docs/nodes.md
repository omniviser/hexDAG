# Nodes Reference

This page documents all registered nodes in HexDAG.

## Overview

Total nodes: **14**

| Alias | Module Path |
|-------|-------------|
| `agent_node` | `hexdag.stdlib.nodes.agent_node.ReActAgentNode` |
| `api_call_node` | `hexdag.stdlib.nodes.api_call_node.ApiCallNode` |
| `checkpoint_node` | `hexdag.stdlib.nodes.checkpoint_node.CheckpointNode` |
| `composite_node` | `hexdag.stdlib.nodes.composite_node.CompositeNode` |
| `data_node` | `hexdag.stdlib.nodes.data_node.DataNode` |
| `expression_node` | `hexdag.stdlib.nodes.expression_node.ExpressionNode` |
| `function_node` | `hexdag.stdlib.nodes.function_node.FunctionNode` |
| `llm_node` | `hexdag.stdlib.nodes.llm_node.LLMNode` |
| `re_act_agent_node` | `hexdag.stdlib.nodes.agent_node.ReActAgentNode` |
| `service_call_node` | `hexdag.stdlib.nodes.service_call_node.ServiceCallNode` |
| `static_node` | `hexdag.stdlib.nodes.data_node.DataNode` |
| `tool_call_node` | `hexdag.stdlib.nodes.tool_call_node.ToolCallNode` |
| `transition` | `hexdag.stdlib.nodes.transition_node.TransitionNode` |
| `transition_node` | `hexdag.stdlib.nodes.transition_node.TransitionNode` |

---

### `agent_node`

**Module:** `hexdag.stdlib.nodes.agent_node.ReActAgentNode`

Multi-step reasoning agent.

**Parameters:**

- **`name`**: `str`
- **`main_prompt`**: `str | orchestration.prompt.template.PromptTemplate | orchestration.prompt.template.ChatPromptTemplate | orchestration.prompt.template.ChatFewShotTemplate | orchestration.prompt.template.FewShotPromptTemplate`
- **`continuation_prompts`**: `dict[str, str | orchestration.prompt.template.PromptTemplate | orchestration.prompt.template.ChatPromptTemplate | orchestration.prompt.template.ChatFewShotTemplate | orchestration.prompt.template.FewShotPromptTemplate] | None` = None
- **`output_schema`**: `dict[str, type] | type[pydantic.main.BaseModel] | None` = None
- **`config`**: `hexdag.stdlib.nodes.agent_node.AgentConfig | None` = None
- **`deps`**: `list[str] | None` = None
- **`kwargs`**: `Any`

**YAML Usage:**

```yaml
nodes:
  - kind: agent_node
    metadata:
      name: agent
    spec:
      main_prompt: "Your prompt here with {{variables}}"
    dependencies: []
```

---

### `api_call_node`

**Module:** `hexdag.stdlib.nodes.api_call_node.ApiCallNode`

Declarative HTTP call node for REST API integration.

**Parameters:**

- **`name`**: `str`
- **`method`**: `HttpMethod` = 'GET'
- **`url`**: `str` = ''
- **`headers`**: `dict[str, str] | None` = None
- **`params`**: `dict[str, Any] | None` = None
- **`json_body`**: `dict[str, Any] | None` = None
- **`port`**: `str` = 'api_call'
- **`output_schema`**: `dict[str, Any] | type[BaseModel] | None` = None
- **`deps`**: `list[str] | None` = None
- **`kwargs`**: `Any`

**YAML Usage:**

```yaml
nodes:
  - kind: api_call_node
    metadata:
      name: api_call
    spec:
      # See parameters above
    dependencies: []
```

---

### `checkpoint_node`

**Module:** `hexdag.stdlib.nodes.checkpoint_node.CheckpointNode`

Declarative mid-pipeline checkpoint save/restore node.

**Parameters:**

- **`name`**: `str`
- **`action`**: `Literal['save', 'restore']`
- **`run_id`**: `str`
- **`keys`**: `list[str] | None` = None
- **`deps`**: `list[str] | None` = None
- **`kwargs`**: `Any`

**YAML Usage:**

```yaml
nodes:
  - kind: checkpoint_node
    metadata:
      name: checkpoint
    spec:
      run_id: "value"
    dependencies: []
```

---

### `composite_node`

**Module:** `hexdag.stdlib.nodes.composite_node.CompositeNode`

Unified control flow node supporting while, for-each, times, if-else, switch.

**Parameters:**

- **`name`**: `str`
- **`mode`**: `Literal['while', 'for-each', 'times', 'if-else', 'switch']`
- **`body`**: `str | list[dict[str, Any]] | collections.abc.Callable[..., Any] | None` = None
- **`body_pipeline`**: `str | None` = None
- **`condition`**: `str | None` = None
- **`items`**: `str | None` = None
- **`item_var`**: `str` = 'item'
- **`index_var`**: `str` = 'index'
- **`count`**: `int | None` = None
- **`branches`**: `list[dict[str, Any]] | None` = None
- **`else_body`**: `str | list[dict[str, Any]] | None` = None
- **`else_action`**: `str | None` = None
- **`initial_state`**: `dict[str, Any] | None` = None
- **`state_update`**: `dict[str, str] | None` = None
- **`max_iterations`**: `int` = 100
- **`concurrency`**: `int` = 1
- **`collect`**: `Literal['list', 'last', 'first', 'dict', 'reduce']` = 'list'
- **`key_field`**: `str | None` = None
- **`reducer`**: `str | None` = None
- **`error_handling`**: `Literal['fail_fast', 'continue', 'collect']` = 'fail_fast'
- **`max_concurrent_nodes`**: `int` = 10
- **`strict_validation`**: `bool` = False
- **`default_node_timeout`**: `float | None` = None
- **`deps`**: `list[str] | None` = None
- **`input_mapping`**: `dict[str, str] | None` = None
- **`kwargs`**: `Any`

**YAML Usage:**

```yaml
nodes:
  - kind: composite_node
    metadata:
      name: composite
    spec:
      # See parameters above
    dependencies: []
```

---

### `data_node`

**Module:** `hexdag.stdlib.nodes.data_node.DataNode`

Static data node factory that returns constant output.

**Parameters:**

- **`name`**: `str`
- **`output`**: `dict[str, Any]`
- **`deps`**: `list[str] | None` = None
- **`kwargs`**: `Any`

**YAML Usage:**

```yaml
nodes:
  - kind: data_node
    metadata:
      name: data
    spec:
      output: "value"
    dependencies: []
```

---

### `expression_node`

**Module:** `hexdag.stdlib.nodes.expression_node.ExpressionNode`

Node factory for computing values using safe AST-based expressions.

**Parameters:**

- **`name`**: `str`
- **`expressions`**: `dict[str, str] | None` = None
- **`input_mapping`**: `dict[str, str] | None` = None
- **`output_fields`**: `list[str] | None` = None
- **`deps`**: `list[str] | None` = None
- **`merge_strategy`**: `Optional[Literal['dict', 'list', 'first', 'last', 'reduce']]` = None
- **`reducer`**: `str | collections.abc.Callable[[list[Any]], Any] | None` = None
- **`extract_field`**: `str | None` = None
- **`kwargs`**: `Any`

**YAML Usage:**

```yaml
nodes:
  - kind: expression_node
    metadata:
      name: expression
    spec:
      # See parameters above
    dependencies: []
```

---

### `function_node`

**Module:** `hexdag.stdlib.nodes.function_node.FunctionNode`

Simple factory for creating function-based nodes with optional Pydantic validation.

**Parameters:**

- **`name`**: `str`
- **`fn`**: `collections.abc.Callable[..., Any] | str`
- **`input_schema`**: `dict[str, Any] | type[pydantic.main.BaseModel] | None` = None
- **`output_schema`**: `dict[str, Any] | type[pydantic.main.BaseModel] | None` = None
- **`deps`**: `list[str] | None` = None
- **`input_mapping`**: `dict[str, str] | None` = None
- **`unpack_input`**: `bool` = False
- **`kwargs`**: `Any`

**YAML Usage:**

```yaml
nodes:
  - kind: function_node
    metadata:
      name: function
    spec:
      fn: "module.function_name"
    dependencies: []
```

---

### `llm_node`

**Module:** `hexdag.stdlib.nodes.llm_node.LLMNode`

Unified LLM node — prompt building, API calls, and structured output.

**Parameters:**

- **`name`**: `str`
- **`human_message`**: `str | None` = None
- **`system_message`**: `str | None` = None
- **`examples`**: `list[dict[str, Any]] | None` = None
- **`conversation`**: `str | list[Any] | None` = None
- **`output_schema`**: `dict[str, Any] | type[BaseModel] | None` = None
- **`deps`**: `list[str] | None` = None
- **`prompt_template`**: `PromptInput | str | None` = None
- **`system_prompt`**: `str | None` = None
- **`parse_json`**: `bool` = False
- **`template`**: `PromptInput | str | None` = None
- **`kwargs`**: `Any`

**YAML Usage:**

```yaml
nodes:
  - kind: llm_node
    metadata:
      name: llm
    spec:
      # See parameters above
    dependencies: []
```

---

### `re_act_agent_node`

**Module:** `hexdag.stdlib.nodes.agent_node.ReActAgentNode`

Multi-step reasoning agent.

**Parameters:**

- **`name`**: `str`
- **`main_prompt`**: `str | orchestration.prompt.template.PromptTemplate | orchestration.prompt.template.ChatPromptTemplate | orchestration.prompt.template.ChatFewShotTemplate | orchestration.prompt.template.FewShotPromptTemplate`
- **`continuation_prompts`**: `dict[str, str | orchestration.prompt.template.PromptTemplate | orchestration.prompt.template.ChatPromptTemplate | orchestration.prompt.template.ChatFewShotTemplate | orchestration.prompt.template.FewShotPromptTemplate] | None` = None
- **`output_schema`**: `dict[str, type] | type[pydantic.main.BaseModel] | None` = None
- **`config`**: `hexdag.stdlib.nodes.agent_node.AgentConfig | None` = None
- **`deps`**: `list[str] | None` = None
- **`kwargs`**: `Any`

**YAML Usage:**

```yaml
nodes:
  - kind: re_act_agent_node
    metadata:
      name: re_act_agent
    spec:
      main_prompt: "Your prompt here with {{variables}}"
    dependencies: []
```

---

### `service_call_node`

**Module:** `hexdag.stdlib.nodes.service_call_node.ServiceCallNode`

Call a ``@step`` method on a Service as a deterministic DAG node.

**Parameters:**

- **`name`**: `str`
- **`service`**: `str`
- **`method`**: `str`
- **`deps`**: `list[str] | None` = None
- **`kwargs`**: `Any`

**YAML Usage:**

```yaml
nodes:
  - kind: service_call_node
    metadata:
      name: service_call
    spec:
      service: "value"
      method: "value"
    dependencies: []
```

---

### `static_node`

**Module:** `hexdag.stdlib.nodes.data_node.DataNode`

Static data node factory that returns constant output.

**Parameters:**

- **`name`**: `str`
- **`output`**: `dict[str, Any]`
- **`deps`**: `list[str] | None` = None
- **`kwargs`**: `Any`

**YAML Usage:**

```yaml
nodes:
  - kind: static_node
    metadata:
      name: static
    spec:
      output: "value"
    dependencies: []
```

---

### `tool_call_node`

**Module:** `hexdag.stdlib.nodes.tool_call_node.ToolCallNode`

Execute a single tool call as a FunctionNode.

**Parameters:**

- **`name`**: `str`
- **`tool_name`**: `str`
- **`arguments`**: `dict[str, Any] | None` = None
- **`tool_call_id`**: `str | None` = None
- **`deps`**: `list[str] | None` = None
- **`kwargs`**: `Any`

**YAML Usage:**

```yaml
nodes:
  - kind: tool_call_node
    metadata:
      name: tool_call
    spec:
      tool_name: "value"
    dependencies: []
```

---

### `transition`

**Module:** `hexdag.stdlib.nodes.transition_node.TransitionNode`

Factory for entity state transition nodes.

**Parameters:**

- **`name`**: `str`
- **`entity`**: `str`
- **`entity_id`**: `str | None` = None
- **`to_state`**: `str`
- **`reason`**: `str | None` = None
- **`kwargs`**: `Any`

**YAML Usage:**

```yaml
nodes:
  - kind: transition
    metadata:
      name: transition
    spec:
      entity: "value"
      to_state: "value"
    dependencies: []
```

---

### `transition_node`

**Module:** `hexdag.stdlib.nodes.transition_node.TransitionNode`

Factory for entity state transition nodes.

**Parameters:**

- **`name`**: `str`
- **`entity`**: `str`
- **`entity_id`**: `str | None` = None
- **`to_state`**: `str`
- **`reason`**: `str | None` = None
- **`kwargs`**: `Any`

**YAML Usage:**

```yaml
nodes:
  - kind: transition_node
    metadata:
      name: transition
    spec:
      entity: "value"
      to_state: "value"
    dependencies: []
```

---
