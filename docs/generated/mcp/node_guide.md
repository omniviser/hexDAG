# Creating Custom Nodes in hexDAG

## Overview

Nodes are the building blocks of hexDAG pipelines. Each node performs a specific
task and can be connected to other nodes via dependencies.

## Quick Start

### Using FunctionNode (Simplest)

Reference any Python function by module path:

```yaml
- kind: function_node
  metadata:
    name: my_processor
  spec:
    fn: mycompany.processors.process_data
  dependencies: []
```

```python
# mycompany/processors.py
def process_data(input_data: dict) -> dict:
    """Your processing logic."""
    return {"result": input_data["value"] * 2}
```

## Available Node Types

### ConditionalNode

Multi-branch conditional router for workflow control flow

**Kind**: `conditional_node`

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `branches` | `array` | Yes | List of condition branches evaluated in order |
| `else_action` | `string` | No | Default action if no branch conditions match |
| `tie_break` | `string` | No | Strategy for handling multiple matching branches |

**Example:**
```yaml
kind: conditional_node
metadata:
  name: my_conditional_node
spec:
  branches: []
  tie_break: first_true
```

### DataNode

Static data node returning constant output. Useful for terminal nodes like rejection actions or static configuration.

**Kind**: `data_node`

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `output` | `object` | Yes | Static output data to return. Can be any JSON-seri... |

**Example:**
```yaml
kind: data_node
metadata:
  name: my_data_node
spec:
  output: {}
```

### FunctionNode

Simple factory for creating function-based nodes with optional Pydantic validation.

**Kind**: `function_node`

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | `str` | Yes | Node name |
| `fn` | `collections.abc.Callable[..., typing.Any] | str` | Yes | Function to execute (callable or module path strin... |
| `input_schema` | `dict[str, typing.Any] | type[pydantic.main.BaseModel] | None` | No | Input schema for validation (if None, inferred fro... |
| `output_schema` | `dict[str, typing.Any] | type[pydantic.main.BaseModel] | None` | No | Output schema for validation (if None, inferred fr... |
| `deps` | `list[str] | None` | No | List of dependency node names |
| `input_mapping` | `dict[str, str] | None` | No | Optional field mapping dict {target_field: "source... |
| `unpack_input` | `bool` | No | If True, unpack input_mapping fields as individual... |

**Example:**
```yaml
- kind: function_node
  metadata:
    name: my_function
  spec:
    # Add configuration here
  dependencies: []
```

### LLMNode

Unified LLM node - prompt building, API calls, and optional parsing.

**Kind**: `l_l_m_node`

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | `str` | Yes | Node name (must be unique in the graph) |
| `prompt_template` | `PromptInput | str | None` | No | Template for the user prompt. Supports Jinja2-styl... |
| `output_schema` | `dict[str, Any] | type[BaseModel] | None` | No | Expected output schema for structured output. If p... |
| `system_prompt` | `str | None` | No | System message to prepend to the conversation. |
| `parse_json` | `bool` | No | If True, parse the LLM response as JSON and valida... |
| `parse_strategy` | `str` | No |  |
| `deps` | `list[str] | None` | No | List of dependency node names. |
| `template` | `PromptInput | str | None` | No |  |

**Example:**
```yaml
- kind: l_l_m_node
  metadata:
    name: my_l_l_m
  spec:
    # Add configuration here
  dependencies: []
```

### LoopNode

Loop control node for iterative processing

**Kind**: `loop_node`

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `while_condition` | `string` | Yes | Module path to condition function: (data, state) -... |
| `body` | `string` | Yes | Module path to body function: (data, state) -> Any |
| `max_iterations` | `integer` | No | Maximum number of iterations before stopping |
| `collect_mode` | `string` | No | How to collect results: last value, all values, or... |
| `initial_state` | `object` | No | Initial state dict passed to first iteration |
| `iteration_key` | `string` | No | Key name for current iteration number in state |

**Example:**
```yaml
kind: loop_node
metadata:
  name: my_loop_node
spec:
  while_condition: value
  body: value
  max_iterations: 100
  collect_mode: last
  initial_state: {}
  iteration_key: loop_iteration
```

### PortCallNode

Execute a method on a configured port/adapter.

**Kind**: `port_call_node`

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | `str` | Yes | Node name (must be unique within the pipeline) |
| `port` | `str` | Yes | Name of the port to call (e.g., "database", "llm",... |
| `method` | `str` | Yes | Method name to invoke on the port |
| `input_mapping` | `dict[str, str] | None` | No | Mapping of method parameter names to data sources.... |
| `fallback` | `Any` | No | Value to return if the port is not available |
| `has_fallback` | `bool` | No | Set to True to enable fallback behavior (allows No... |
| `output_schema` | `dict[str, Any] | type[BaseModel] | None` | No | Optional schema for validating/structuring the out... |
| `deps` | `list[str] | None` | No | List of dependency node names for execution orderi... |

**Example:**
```yaml
- kind: port_call_node
  metadata:
    name: my_port_call
  spec:
    # Add configuration here
  dependencies: []
```

### ReActAgentNode

Multi-step reasoning agent.

**Kind**: `re_act_agent_node`

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | `str` | Yes | Agent name |
| `main_prompt` | `str | hexdag.core.orchestration.prompt.template.PromptTemplate | hexdag.builtin.prompts.base.ChatPromptTemplate | hexdag.builtin.prompts.base.ChatFewShotTemplate | hexdag.builtin.prompts.base.FewShotPromptTemplate` | Yes | Initial reasoning prompt |
| `continuation_prompts` | `dict[str, str | hexdag.core.orchestration.prompt.template.PromptTemplate | hexdag.builtin.prompts.base.ChatPromptTemplate | hexdag.builtin.prompts.base.ChatFewShotTemplate | hexdag.builtin.prompts.base.FewShotPromptTemplate] | None` | No | Phase-specific prompts |
| `output_schema` | `dict[str, type] | type[pydantic.main.BaseModel] | None` | No | Custom output schema for tool_end results |
| `config` | `hexdag.builtin.nodes.agent_node.AgentConfig | None` | No | Agent configuration |
| `deps` | `list[str] | None` | No | Dependencies |

**Example:**
```yaml
- kind: re_act_agent_node
  metadata:
    name: my_re_act_agent
  spec:
    # Add configuration here
  dependencies: []
```

### ToolCallNode

Execute a single tool call as a FunctionNode.

**Kind**: `tool_call_node`

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | `str` | Yes | Node name (should be unique) |
| `tool_name` | `str` | Yes | Full module path to the tool function (e.g., 'mymo... |
| `arguments` | `dict[str, typing.Any] | None` | No | Arguments to pass to the tool (default: {}) |
| `tool_call_id` | `str | None` | No | Optional ID for tracking (from LLM tool calls) |
| `deps` | `list[str] | None` | No | Dependencies (typically the LLM node that requeste... |

**Example:**
```yaml
- kind: tool_call_node
  metadata:
    name: my_tool_call
  spec:
    # Add configuration here
  dependencies: []
```

## Creating Custom Nodes

```python
from hexdag.builtin.nodes import BaseNodeFactory
from hexdag.core.domain.dag import NodeSpec

class CustomProcessorNode(BaseNodeFactory):
    """Custom node for specialized processing."""

    def __call__(
        self,
        name: str,
        threshold: float = 0.5,
        **kwargs
    ) -> NodeSpec:
        async def process_fn(input_data: dict) -> dict:
            if input_data.get("score", 0) > threshold:
                return {"status": "pass"}
            return {"status": "fail"}

        return NodeSpec(
            name=name,
            fn=process_fn,
            deps=frozenset(kwargs.get("deps", [])),
        )
```

## Best Practices

1. **Async Functions**: Use `async def` for the node function
2. **Immutable**: Don't modify input_data; return new dict
3. **Type Hints**: Add types for better IDE support
4. **Docstrings**: Document purpose and parameters
