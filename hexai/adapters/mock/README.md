# Mock Adapters

Mock adapter implementations for testing and development without external dependencies.

## Loading Mock Adapters

Mock adapters are **not loaded by default**. To use them, you have two options:

### Option 1: Load via Configuration File

```python
from hexai.core.bootstrap import bootstrap_registry

# Load mock adapters explicitly
bootstrap_registry("hexai/adapters/mock/hexdag.toml")
```

### Option 2: Direct Import

```python
from hexai.adapters.mock.mock_llm import MockLLM
from hexai.adapters.mock.mock_database import MockDatabaseAdapter
from hexai.adapters.mock.mock_tool_router import MockToolRouter

# Create instances directly
llm = MockLLM()
db = MockDatabaseAdapter()
router = MockToolRouter()
```

## Available Mock Adapters

### MockLLM
Simulates LLM responses without API calls.

```python
from hexai.core.ports.llm import Message

llm = registry.get("mock_llm", namespace="plugin")
messages = [Message(role="user", content="Hello!")]
response = await llm.aresponse(messages)
```

### MockDatabaseAdapter
Provides sample e-commerce data for testing database operations.

```python
db = registry.get("mock_database", namespace="plugin")
schemas = await db.aget_table_schemas()
results = await db.aexecute_query("SELECT * FROM customers LIMIT 5")
```

### MockToolRouter
Simulates tool execution with predefined responses.

```python
router = registry.get("mock_tool_router", namespace="plugin")
tools = router.get_available_tools()
result = await router.acall_tool("calculate", {"expression": "2 + 2"})
```

## Configuration

Mock adapters can be configured via the `hexdag.toml` file:

```toml
[settings.mock_llm]
responses = ["Custom response 1", "Custom response 2"]
delay_seconds = 0.1

[settings.mock_database]
enable_sample_data = true
delay_seconds = 0.0

[settings.mock_tool_router]
available_tools = ["search", "calculate", "get_weather"]
raise_on_unknown_tool = true
```

## Use Cases

- **Unit Testing**: Test workflows without external dependencies
- **Development**: Rapid prototyping without API keys or database setup
- **CI/CD**: Reliable, deterministic tests in pipelines
- **Demos**: Showcase functionality without infrastructure

## Example

See `examples/22_mock_adapters_demo.py` for a complete demonstration of using mock adapters in workflows.
