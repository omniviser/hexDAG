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

### CompositeNode

Unified control flow node supporting while, for-each, times, if-else, switch.

**Kind**: `composite_node`

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | `str` | Yes | Node name (unique identifier in the pipeline) |
| `mode` | `Literal['while', 'for-each', 'times', 'if-else', 'switch']` | Yes | Control flow mode: while, for-each, times, if-else... |
| `body` | `str | list[dict[str, typing.Any]] | collections.abc.Callable[..., typing.Any] | None` | No | Body to execute. Can be: - Module path string (e.g... |
| `body_pipeline` | `str | None` | No | Path to external pipeline YAML file |
| `condition` | `str | None` | No | Condition expression for while, if-else, or switch... |
| `items` | `str | None` | No | Expression resolving to iterable for for-each mode |
| `item_var` | `str` | No | Variable name for current item (default: "item") |
| `index_var` | `str` | No | Variable name for current index (default: "index") |
| `count` | `int | None` | No | Number of iterations for times mode |
| `branches` | `list[dict[str, typing.Any]] | None` | No | List of condition branches for switch mode |
| `else_body` | `str | list[dict[str, typing.Any]] | None` | No | Body for else branch (if-else, switch with inline ... |
| `else_action` | `str | None` | No | Action label for else branch (switch routing mode) |
| `initial_state` | `dict[str, typing.Any] | None` | No | Initial state dict for while mode |
| `state_update` | `dict[str, str] | None` | No | State update expressions for while mode |
| `max_iterations` | `int` | No | Safety limit for while loops (default: 100) |
| `concurrency` | `int` | No | Max concurrent iterations for for-each/times (defa... |
| `collect` | `Literal['list', 'last', 'first', 'dict', 'reduce']` | No | Result collection mode (default: "list") |
| `key_field` | `str | None` | No | Field to use as key for dict collection |
| `reducer` | `str | None` | No | Module path to reducer function for reduce collect... |
| `error_handling` | `Literal['fail_fast', 'continue', 'collect']` | No | Error handling strategy (default: "fail_fast") |
| `max_concurrent_nodes` | `int` | No |  |
| `strict_validation` | `bool` | No |  |
| `default_node_timeout` | `float | None` | No |  |
| `deps` | `list[str] | None` | No | Dependency node names |
| `input_mapping` | `dict[str, str] | None` | No | Field extraction mapping for orchestrator |

**Example:**
```yaml
- kind: composite_node
  metadata:
    name: my_composite
  spec:
    # Add configuration here
  dependencies: []
```

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

Static data node factory that returns constant output.

**Kind**: `data_node`

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | `str` | Yes | Node name (must be unique within the pipeline) |
| `output` | `dict[str, Any]` | Yes | Output data to return. Values can be: - Static val... |
| `deps` | `list[str] | None` | No | List of dependency node names for execution orderi... |

**Example:**
```yaml
- kind: data_node
  metadata:
    name: my_data
  spec:
    # Add configuration here
  dependencies: []
```

### ExpressionNode

Node factory for computing values using safe AST-based expressions.

**Kind**: `expression_node`

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | `str` | Yes | Node name (unique identifier in the pipeline) |
| `expressions` | `dict[str, str] | None` | No | Mapping of {variable_name: expression_string}. Exp... |
| `input_mapping` | `dict[str, str] | None` | No | Field extraction mapping {local_name: "source_node... |
| `output_fields` | `list[str] | None` | No | Fields to include in output dict. If None, all com... |
| `deps` | `list[str] | None` | No | Dependency node names (for DAG ordering) |
| `merge_strategy` | `Literal['dict', 'list', 'first', 'last', 'reduce'] | None` | No | Strategy for merging multiple dependency outputs: ... |
| `reducer` | `str | collections.abc.Callable[[list[typing.Any]], typing.Any] | None` | No | Module path (e.g., "statistics.mean") or callable ... |
| `extract_field` | `str | None` | No | Field to extract from each dependency result befor... |

**Example:**
```yaml
- kind: expression_node
  metadata:
    name: my_expression
  spec:
    # Add configuration here
  dependencies: []
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
| `parse_strategy` | `str` | No | JSON parsing strategy: "json", "json_in_markdown",... |
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
