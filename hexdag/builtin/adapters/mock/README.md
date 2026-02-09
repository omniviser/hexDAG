# Mock Adapters

Mock adapter implementations for testing and development without external dependencies.

## Loading Mock Adapters

Mock adapters can be imported directly from the builtin package:

```python
from hexdag.builtin.adapters.mock.mock_llm import MockLLM
from hexdag.builtin.adapters.mock.mock_database import MockDatabaseAdapter
from hexdag.builtin.adapters.mock.mock_tool_router import MockToolRouter

# Create instances directly
llm = MockLLM()
db = MockDatabaseAdapter()
router = MockToolRouter()
```

Or reference them by their full module path in YAML pipelines:

```yaml
spec:
  ports:
    llm:
      adapter: hexdag.builtin.adapters.mock.MockLLM
      config:
        responses: ["Custom response"]
```

## Available Mock Adapters

### MockLLM
Simulates LLM responses without API calls.

```python
from hexdag.builtin.adapters.mock.mock_llm import MockLLM
from hexdag.core.ports.llm import Message

llm = MockLLM(responses=["Hello! How can I help?"])
messages = [Message(role="user", content="Hello!")]
response = await llm.aresponse(messages)
```

### MockDatabaseAdapter
Provides sample e-commerce data for testing database operations.

```python
from hexdag.builtin.adapters.mock.mock_database import MockDatabaseAdapter

db = MockDatabaseAdapter()
schemas = await db.aget_table_schemas()
results = await db.aexecute_query("SELECT * FROM customers LIMIT 5")
```

### MockToolRouter
Simulates tool execution with predefined responses.

```python
from hexdag.builtin.adapters.mock.mock_tool_router import MockToolRouter

router = MockToolRouter()
tools = router.get_available_tools()
result = await router.acall_tool("calculate", {"expression": "2 + 2"})
```

## Configuration in YAML

Configure mock adapters directly in your YAML pipeline:

```yaml
spec:
  ports:
    llm:
      adapter: hexdag.builtin.adapters.mock.MockLLM
      config:
        responses: ["Response 1", "Response 2"]
        delay_seconds: 0.1

    database:
      adapter: hexdag.builtin.adapters.mock.MockDatabaseAdapter
      config:
        enable_sample_data: true
        delay_seconds: 0.0

    tool_router:
      adapter: hexdag.builtin.adapters.mock.MockToolRouter
      config:
        available_tools: ["search", "calculate", "get_weather"]
        raise_on_unknown_tool: true
```

## Use Cases

- **Unit Testing**: Test workflows without external dependencies
- **Development**: Rapid prototyping without API keys or database setup
- **CI/CD**: Reliable, deterministic tests in pipelines
- **Demos**: Showcase functionality without infrastructure

## Example

See `examples/22_mock_adapters_demo.py` for a complete demonstration of using mock adapters in workflows.
