# 🔷 hexDAG's Hexagonal Architecture

## Why Hexagonal Architecture?

hexDAG embraces **Hexagonal Architecture** (Ports and Adapters) to create a truly flexible, testable, and maintainable AI orchestration framework. This architectural pattern ensures your business logic remains pure and independent from external dependencies.

## 🎯 Core Philosophy

```
         ┌─────────────────────────────────┐
         │                                   │
         │         Business Logic            │
         │     (DAG, Nodes, Orchestrator)    │
         │                                   │
         │  ┌───────────────────────────┐   │
         │  │                           │   │
    ┌────┼──│    Domain Core            │───┼────┐
    │    │  │   - DirectedGraph         │   │    │
    │    │  │   - NodeSpec              │   │    │
    │    │  │   - Orchestration Rules   │   │    │
    │    │  └───────────────────────────┘   │    │
    │    │                                   │    │
    │    └─────────────────────────────────┘    │
    │                                             │
    ▼                                             ▼
┌─────────┐                                   ┌─────────┐
│  PORTS  │                                   │  PORTS  │
├─────────┤                                   ├─────────┤
│   LLM   │                                   │ Memory  │
│Database │                                   │  Tools  │
│ Events  │                                   │ Config  │
└─────────┘                                   └─────────┘
    │                                             │
    ▼                                             ▼
┌─────────────────┐                   ┌─────────────────┐
│    ADAPTERS     │                   │    ADAPTERS     │
├─────────────────┤                   ├─────────────────┤
│ OpenAI          │                   │ Redis           │
│ Anthropic       │                   │ PostgreSQL      │
│ Local LLM       │                   │ In-Memory       │
│ Mock (Testing)  │                   │ Mock (Testing)  │
└─────────────────┘                   └─────────────────┘
```

## 🔌 Ports: The Contract

Ports define **what** your application needs, not **how** it gets it:

```python
# Pure interface - no implementation details
class LLMPort(Protocol):
    """Contract for language model interactions."""

    async def agenerate(
        self,
        messages: list[Message],
        **kwargs
    ) -> LLMResponse:
        """Generate response from messages."""
        ...

class MemoryPort(Protocol):
    """Contract for memory operations."""

    async def astore(self, key: str, value: Any) -> None:
        """Store a value."""
        ...

    async def aretrieve(self, key: str) -> Any:
        """Retrieve a value."""
        ...
```

## 🔄 Adapters: The Implementation

Adapters implement **how** to fulfill the port contracts. They are plain Python classes that implement the port interface:

```python
class OpenAIAdapter:
    """OpenAI implementation of LLM port.

    Referenced in YAML as: hexdag.stdlib.adapters.openai.OpenAIAdapter
    """

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model

    async def agenerate(self, messages, **kwargs):
        # Actual OpenAI API calls
        return await self.client.chat.completions.create(...)


class MockLLMAdapter:
    """Test implementation for fast, deterministic testing.

    Referenced in YAML as: hexdag.stdlib.adapters.mock.MockLLM
    """

    async def agenerate(self, messages, **kwargs):
        # Return predictable test responses
        return LLMResponse(content="Test response")
```

## 💡 Benefits in Practice

### 1. **Swap Without Breaking**

```yaml
# Production configuration
adapters:
  llm: openai
  memory: redis
  database: postgresql

# Testing configuration
adapters:
  llm: mock
  memory: in_memory
  database: sqlite
```

Your entire pipeline code remains **unchanged**. Only configuration differs.

### 2. **Test at Lightning Speed**

```python
# Test runs with mock adapters - instant, no API calls
async def test_complex_workflow():
    graph = build_graph()
    orchestrator = Orchestrator(
        llm_adapter=MockLLMAdapter(),
        memory_adapter=InMemoryAdapter()
    )
    result = await orchestrator.run(graph)
    assert result.success
```

### 3. **Add New Providers Easily**

```python
# Adding a new LLM provider takes minutes, not days
class CohereAdapter:
    """Cohere implementation of LLM port.

    Referenced by full module path in YAML:
    adapters:
      llm:
        adapter: mypackage.adapters.CohereAdapter
    """

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def agenerate(self, messages, **kwargs):
        # Your Cohere implementation
        pass
```

### 4. **Business Logic Stays Pure**

```python
# Your node doesn't know or care which LLM it's using
class AnalysisNode:
    async def __call__(self, context):
        # Pure business logic
        response = await context.llm.agenerate(
            messages=[{"role": "user", "content": context.input}]
        )
        return self.process_response(response)
```

## 🏗️ Architecture Layers

### Domain Layer (Core)
- **DirectedGraph**: Pure graph structure and validation
- **NodeSpec**: Node definitions and dependencies
- **Orchestrator**: Execution logic
- **No external dependencies**

### Application Layer
- **YamlPipelineBuilder**: Pipeline construction
- **NodeFactories**: Node creation logic
- **Validation**: Type checking and schema validation
- **Uses ports, never adapters directly**

### Infrastructure Layer
- **Adapters**: External service implementations
- **Configuration**: Environment management
- **Logging**: Observability infrastructure
- **Can be completely replaced**

## 🎯 Real-World Impact

### Before Hexagonal Architecture
```python
class ChatNode:
    def __init__(self):
        # Tightly coupled to OpenAI
        self.client = OpenAI(api_key="...")

    async def process(self, input):
        # Hard to test, hard to change
        response = await self.client.chat(...)
        # What if you want to use Claude?
        # What if OpenAI is down?
        # How do you test this?
```

### With Hexagonal Architecture
```python
class ChatNode:
    async def __call__(self, context):
        # Works with ANY LLM adapter
        response = await context.ports.llm.agenerate(
            messages=context.messages
        )
        # Testable, swappable, maintainable
```

## 🚀 Why This Matters

1. **Vendor Independence**: Switch from OpenAI to Anthropic to local LLMs with one line
2. **Test Confidence**: Test your entire pipeline without spending a penny on API calls
3. **Gradual Migration**: Move from SQLite to PostgreSQL when you're ready
4. **Multi-Environment**: Dev, staging, and production with different adapters
5. **Future-Proof**: New AI providers? Just add an adapter

## 📊 Comparison

| Aspect | Traditional Architecture | Hexagonal Architecture |
|--------|-------------------------|------------------------|
| Testing | Slow, expensive (real APIs) | Fast, free (mock adapters) |
| Vendor Lock-in | High (hardcoded deps) | None (swappable adapters) |
| New Provider | Major refactor | Add one adapter class |
| Environment Config | Complex conditionals | Simple adapter swap |
| Business Logic | Mixed with infrastructure | Pure and isolated |

## 🎭 The hexDAG Advantage

hexDAG's hexagonal architecture isn't just theory—it's practical engineering that gives you:

- **10x faster tests** (mock adapters)
- **Zero vendor lock-in** (swap providers anytime)
- **Clean, maintainable code** (separation of concerns)
- **Production confidence** (same code, different adapters)

---

**The result?** You focus on building AI workflows, not wrestling with infrastructure. Your business logic remains pure, your tests run fast, and your system adapts to any provider or service you need.

*This is why hexDAG scales from prototypes to production without rewrites.*
