# Nodes Reference

This page documents all registered nodes in HexDAG.

## Overview

Total nodes: **5**

## Namespace: `core`

### `agent_node`

Multi-step reasoning agent.

    This agent:
    1. Uses loop control internally for iteration control
    2. Implements single-step reasoning logic
    3. Maintains clean agent interface for users
    4. Leverages proven loop control patterns
    5. Supports all agent features (tools, phases, events)

    Architecture:
    ```
    Agent(input) -> Loop -> SingleStep -> Loop -> SingleStep -> ... -> Output
    ```

**Metadata:**

- **Type:** `node`
- **Namespace:** `core`
- **Name:** `agent_node`
- **Subtype:** `agent`

**YAML Usage:**

```yaml
- type: agent
  id: agent
  params:
    main_prompt: "Your prompt here with {{variables}}"  # Required
    continuation_prompts: "Your prompt here with {{variables}}"  # Optional (default: None)
    output_schema: "value"  # Optional (default: None)
  depends_on: []  # List of upstream node IDs
```

**Parameters:**

- **`main_prompt`** (`str | hexdag.core.orchestration.prompt.template.PromptTemplate | hexdag.core.orchestration.prompt.template.ChatPromptTemplate | hexdag.core.orchestration.prompt.template.ChatFewShotTemplate | hexdag.core.orchestration.prompt.template.FewShotPromptTemplate`, **required**)
- **`continuation_prompts`** (`dict[str, str | hexdag.core.orchestration.prompt.template.PromptTemplate | hexdag.core.orchestration.prompt.template.ChatPromptTemplate | hexdag.core.orchestration.prompt.template.ChatFewShotTemplate | hexdag.core.orchestration.prompt.template.FewShotPromptTemplate] | None`, optional, default: `None`)
- **`output_schema`** (`dict[str, type] | type[pydantic.main.BaseModel] | None`, optional, default: `None`)

---

### `conditional_node`

Factory class for creating conditional routing nodes.

    Conditional nodes are dynamic - condition_key and actions are passed via __call__()
    parameters at node creation time. No static Config needed (follows YAGNI principle).

**Metadata:**

- **Type:** `node`
- **Namespace:** `core`
- **Name:** `conditional_node`
- **Subtype:** `conditional`

**YAML Usage:**

```yaml
- type: conditional
  id: conditional
  params:
    condition_key: "{{expression}}"  # Optional (default: should_continue)
    true_action: "value"  # Optional (default: continue)
    false_action: "value"  # Optional (default: proceed)
  depends_on: []  # List of upstream node IDs
```

**Parameters:**

- **`condition_key`** (`str`, optional, default: `should_continue`)
- **`true_action`** (`str`, optional, default: `continue`)
- **`false_action`** (`str`, optional, default: `proceed`)

---

### `function_node`

Simple factory for creating function-based nodes with optional Pydantic validation.

    Function nodes are highly dynamic - the function itself defines configuration via its
    signature and parameters. No static Config class needed (follows YAGNI principle).
    All configuration is passed dynamically through __call__() parameters.

**Metadata:**

- **Type:** `node`
- **Namespace:** `core`
- **Name:** `function_node`
- **Subtype:** `function`

**YAML Usage:**

```yaml
- type: function
  id: function
  params:
    fn: "module.function_name"  # Required
    input_schema: "value"  # Optional (default: None)
    output_schema: "value"  # Optional (default: None)
    input_mapping: "value"  # Optional (default: None)
  depends_on: []  # List of upstream node IDs
```

**Parameters:**

- **`fn`** (`collections.abc.Callable[..., Any]`, **required**)
- **`input_schema`** (`dict[str, Any] | type[pydantic.main.BaseModel] | None`, optional, default: `None`)
- **`output_schema`** (`dict[str, Any] | type[pydantic.main.BaseModel] | None`, optional, default: `None`)
- **`input_mapping`** (`dict[str, str] | None`, optional, default: `None`)

---

### `llm_node`

Simple factory for creating LLM-based nodes with rich template support.

    Inherits all common LLM functionality from BaseLLMNode. LLM nodes are highly dynamic -
    templates and schemas are passed via __call__() parameters rather than static Config.
    No Config class needed (follows YAGNI principle).

**Metadata:**

- **Type:** `node`
- **Namespace:** `core`
- **Name:** `llm_node`
- **Subtype:** `llm`

**YAML Usage:**

```yaml
- type: llm
  id: llm
  params:
    template: "Your prompt here with {{variables}}"  # Required
    output_schema: "value"  # Optional (default: None)
  depends_on: []  # List of upstream node IDs
```

**Parameters:**

- **`template`** (`str | hexdag.core.orchestration.prompt.template.PromptTemplate | hexdag.core.orchestration.prompt.template.ChatPromptTemplate | hexdag.core.orchestration.prompt.template.ChatFewShotTemplate | hexdag.core.orchestration.prompt.template.FewShotPromptTemplate`, **required**)
- **`output_schema`** (`dict[str, Any] | type[pydantic.main.BaseModel] | None`, optional, default: `None`)

---

### `loop_node`

Factory class for creating loop control nodes with iteration management.

    Loop nodes are dynamic - max_iterations and success_condition are passed via __call__()
    parameters at node creation time. No static Config needed (follows YAGNI principle).

**Metadata:**

- **Type:** `node`
- **Namespace:** `core`
- **Name:** `loop_node`
- **Subtype:** `loop`

**YAML Usage:**

```yaml
- type: loop
  id: loop
  params:
    max_iterations: 100  # Optional (default: 3)
    success_condition: "{{expression}}"  # Optional (default: None)
    iteration_key: "value"  # Optional (default: loop_iteration)
  depends_on: []  # List of upstream node IDs
```

**Parameters:**

- **`max_iterations`** (`int`, optional, default: `3`)
- **`success_condition`** (`collections.abc.Callable[[Any], bool] | None`, optional, default: `None`)
- **`iteration_key`** (`str`, optional, default: `loop_iteration`)

---
