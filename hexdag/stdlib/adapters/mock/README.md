# Mock Adapters

Mock adapter implementations for testing and development without external dependencies.

## Loading Mock Adapters

Mock adapters can be imported directly from the builtin package:

```python
from hexdag.stdlib.adapters.mock.mock_llm import MockLLM
from hexdag.stdlib.adapters.mock.mock_database import MockDatabaseAdapter

# Create instances directly
llm = MockLLM()
db = MockDatabaseAdapter()
```

Or reference them by their full module path in YAML pipelines:

```yaml
spec:
  ports:
    llm:
      adapter: hexdag.stdlib.adapters.mock.MockLLM
      config:
        responses: ["Custom response"]
```

## Available Mock Adapters

### MockLLM
Simulates LLM responses without API calls.

```python
from hexdag.stdlib.adapters.mock.mock_llm import MockLLM
from hexdag.kernel.ports.llm import Message

llm = MockLLM(responses=["Hello! How can I help?"])
messages = [Message(role="user", content="Hello!")]
response = await llm.aresponse(messages)
```

### MockDatabaseAdapter
Provides sample e-commerce data for testing database operations.

```python
from hexdag.stdlib.adapters.mock.mock_database import MockDatabaseAdapter

db = MockDatabaseAdapter()
schemas = await db.aget_table_schemas()
results = await db.aexecute_query("SELECT * FROM customers LIMIT 5")
```

### Tool Router for Testing

For tool routing in tests, use the base `ToolRouter` with mock functions:

```python
from hexdag.kernel.ports.tool_router import ToolRouter

router = ToolRouter(tools={
    "search": lambda query="", **kw: {"results": [f"Result for: {query}"]},
    "calculate": lambda expression="", **kw: {"result": str(expression)},
})
tools = router.get_available_tools()
result = await router.acall_tool("calculate", {"expression": "2 + 2"})
```

## Configuration in YAML

Configure mock adapters directly in your YAML pipeline:

```yaml
spec:
  ports:
    llm:
      adapter: hexdag.stdlib.adapters.mock.MockLLM
      config:
        responses: ["Response 1", "Response 2"]
        delay_seconds: 0.1

    database:
      adapter: hexdag.stdlib.adapters.mock.MockDatabaseAdapter
      config:
        enable_sample_data: true
        delay_seconds: 0.0

    tool_router:
      adapter: hexdag.kernel.ports.tool_router.ToolRouter
      config:
        tools: {}
```

## Use Cases

- **Unit Testing**: Test workflows without external dependencies
- **Development**: Rapid prototyping without API keys or database setup
- **CI/CD**: Reliable, deterministic tests in pipelines
- **Demos**: Showcase functionality without infrastructure

## Example

See `examples/22_mock_adapters_demo.py` for a complete demonstration of using mock adapters in workflows.
