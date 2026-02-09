# 🎭 hexDAG Macro System

## Overview

The hexDAG Macro System provides a powerful way to define reusable pipeline patterns that can be invoked from YAML configurations or Python code. Macros are templates that expand at runtime into full DirectedGraph subgraphs, enabling code reuse and simplifying complex workflow definitions.

## Key Concepts

### What is a Macro?

A macro is a reusable pipeline template that:
- Encapsulates common patterns (conversation flows, reasoning chains, tool usage)
- Accepts configuration parameters to customize behavior
- Expands into a complete DAG subgraph at runtime
- Can be invoked from YAML using `macro_invocation` node type
- Supports configuration inheritance and overrides

### Expansion Strategies

hexDAG supports three macro expansion strategies:

1. **STATIC**: Expand once at pipeline build time
2. **DYNAMIC**: Expand at runtime when encountered
3. **LAZY**: Expand only when needed (future)

## Built-in Macros

### ConversationMacro

Multi-turn conversation support with persistent memory:

```yaml
nodes:
  - type: macro_invocation
    id: chat_bot
    macro: conversation
    params:
      system_prompt: "You are a helpful AI assistant"
      memory_adapter: "redis"  # or "in_memory", "sqlite"
      max_turns: 10
      temperature: 0.7
```

**Features:**
- Configurable memory port for conversation history
- System prompt customization
- Turn limits and safety controls
- Automatic context management

### LLMMacro

Simplified LLM invocation with smart defaults:

```yaml
nodes:
  - type: macro_invocation
    id: summarizer
    macro: llm
    params:
      prompt_template: "Summarize this text: {{input_text}}"
      model: "gpt-4"
      temperature: 0.3
      max_tokens: 500
```

**Features:**
- YAML schema conversion
- Template variable support
- Model parameter configuration
- Automatic retry on failures

### ToolMacro

Tool integration for function calling:

```yaml
nodes:
  - type: macro_invocation
    id: calculator
    macro: tool
    params:
      tool_name: "calculator"
      tools:
        - name: add
          description: "Add two numbers"
        - name: multiply
          description: "Multiply two numbers"
```

**Features:**
- Tool registration and discovery
- Input/output validation
- Error handling
- Tool chaining support

### ReasoningAgentMacro

Advanced reasoning patterns for complex decisions:

```yaml
nodes:
  - type: macro_invocation
    id: decision_maker
    macro: reasoning_agent
    params:
      reasoning_strategy: "chain_of_thought"
      max_reasoning_steps: 5
      confidence_threshold: 0.8
```

**Features:**
- Multiple reasoning strategies
- Step-by-step thought process
- Confidence scoring
- Self-correction mechanisms

## Using Macros in YAML

### Basic Invocation

```yaml
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: customer-support
spec:
  nodes:
    - type: macro_invocation
      id: support_agent
      macro: conversation
      params:
        system_prompt: |
          You are a customer support agent.
          Be helpful and professional.
        memory_adapter: "redis"
        max_turns: 20
      depends_on: []

    - type: function
      id: log_conversation
      params:
        fn: save_to_database
      depends_on: [support_agent]
```

### Configuration Override

Override macro defaults at invocation:

```yaml
nodes:
  - type: macro_invocation
    id: creative_writer
    macro: llm
    params:
      # Override default temperature
      temperature: 0.9
      # Override default model
      model: "gpt-4-turbo"
      # Custom prompt
      prompt_template: "Write a creative story about: {{topic}}"
```

### Chaining Macros

Combine multiple macros in a workflow:

```yaml
nodes:
  # First macro: gather information
  - type: macro_invocation
    id: researcher
    macro: reasoning_agent
    params:
      reasoning_strategy: "research"

  # Second macro: have a conversation about findings
  - type: macro_invocation
    id: discussion
    macro: conversation
    params:
      system_prompt: "Discuss the research findings"
    depends_on: [researcher]

  # Third macro: summarize the discussion
  - type: macro_invocation
    id: summarizer
    macro: llm
    params:
      prompt_template: "Summarize: {{discussion.output}}"
    depends_on: [discussion]
```

## Creating Custom Macros

### Python Implementation

```python
from hexdag.core.configurable import ConfigurableMacro, MacroConfig
from pydantic import Field

class CustomMacroConfig(MacroConfig):
    """Configuration for custom macro."""
    custom_param: str = Field(description="Custom parameter")
    another_param: int = Field(default=10)

class CustomMacro(ConfigurableMacro[CustomMacroConfig]):
    """A custom reusable pattern."""

    config_class = CustomMacroConfig

    def expand(self, config: CustomMacroConfig, context) -> DirectedGraph:
        """Expand macro into a DAG subgraph."""
        graph = DirectedGraph()

        # Add nodes based on configuration
        graph.add_node(NodeSpec(
            id=f"{config.id}_step1",
            type="function",
            params={"fn": "process_data"}
        ))

        graph.add_node(NodeSpec(
            id=f"{config.id}_step2",
            type="llm",
            params={"prompt": config.custom_param},
            depends_on=[f"{config.id}_step1"]
        ))

        return graph
```

### Using Custom Macros

Reference macros by their full module path in YAML:

```python
# Your macro is now available in YAML
```

```yaml
nodes:
  - type: macro_invocation
    id: my_custom
    macro: custom_pattern
    params:
      custom_param: "Process this data"
      another_param: 20
```

## Best Practices

### 1. Configuration Design

- Keep macro configurations simple and focused
- Provide sensible defaults
- Document all parameters clearly
- Use Pydantic for validation

### 2. Naming Conventions

- Use descriptive macro names
- Follow snake_case for macro names
- Use clear parameter names

### 3. Error Handling

- Validate inputs early
- Provide helpful error messages
- Handle edge cases gracefully

### 4. Testing

```python
def test_custom_macro():
    config = CustomMacroConfig(
        id="test",
        custom_param="test_value"
    )

    macro = CustomMacro()
    graph = macro.expand(config, {})

    # Verify graph structure
    assert len(graph.nodes) == 2
    assert graph.nodes[0].id == "test_step1"
```

## Advanced Features

### Dynamic Expansion

For runtime flexibility:

```python
class DynamicMacro(ConfigurableMacro):
    """Expands differently based on runtime context."""

    def expand(self, config, context):
        # Access runtime context
        if context.get("mode") == "verbose":
            # Create detailed pipeline
            pass
        else:
            # Create simple pipeline
            pass
```

### Nested Macros

Macros can invoke other macros:

```python
def expand(self, config, context):
    graph = DirectedGraph()

    # Add a macro invocation node
    graph.add_node(NodeSpec(
        id="nested_macro",
        type="macro_invocation",
        params={
            "macro": "llm",
            "prompt_template": "Process: {{input}}"
        }
    ))

    return graph
```

## Performance Considerations

- **STATIC** expansion is fastest (build-time)
- **DYNAMIC** allows runtime flexibility
- Cache expanded graphs when possible
- Consider macro granularity

## Troubleshooting

### Common Issues

1. **Macro not found**: Ensure macro is registered
2. **Invalid configuration**: Check Pydantic validation
3. **Circular dependencies**: Review graph structure
4. **Runtime errors**: Add proper error handling

### Debugging

Enable debug logging:

```python
import logging
logging.getLogger("hexdag.macros").setLevel(logging.DEBUG)
```

## Examples

See working examples:
- [Conversation Bot](../examples/macro_conversation.py)
- [Research Assistant](../examples/macro_research.py)
- [Tool Chain](../examples/macro_tools.py)

## API Reference

- `ConfigurableMacro`: Base class for macros (`hexdag.core.configurable`)
- `MacroConfig`: Base configuration class
- `YamlMacro`: YAML-defined macro class (`hexdag.core.yaml_macro`)

---

For more information, see the [Implementation Guide](IMPLEMENTATION_GUIDE.md) and [Plugin System](PLUGIN_SYSTEM.md) documentation.
