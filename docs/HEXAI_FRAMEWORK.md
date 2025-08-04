# 🔗 hexDAG Framework

> **Enterprise-grade AI agent orchestration framework built on hexagonal architecture with DAG-based execution**

## 🎯 Framework Overview

hexDAG is the core orchestration framework that enables building sophisticated AI agent workflows through declarative configuration. It provides the fundamental building blocks for:

- **AI Agent Orchestration**: Multi-agent coordination with built-in memory and tool support
- **Data Science Workflows**: Seamless integration of AI agents with traditional data processing
- **Real-Time Streaming**: Live updates of agent actions and memory operations to frontends
- **Low-Code Development**: YAML-based declarative configuration for rapid development

## 🎯 Vision & Design Philosophy

hexDAG transforms complex AI workflows into **deterministic, testable, and maintainable** systems by implementing the six pillars:

1. **🔄 Async-First Architecture**: Non-blocking execution for maximum performance
2. **📊 Event-Driven Observability**: Real-time monitoring of agent actions
3. **✅ Pydantic Validation Everywhere**: Type safety at every layer
4. **🏗️ Hexagonal Architecture**: Clean separation of business logic and infrastructure
5. **📝 Composable Declarative Files**: Build complex workflows from simple components
6. **🔀 DAG-Based Orchestration**: Intelligent dependency management and parallelization

### Core Architectural Benefits

- **Deterministic Execution**: Same input always produces same execution order
- **Dependency Injection**: External services provided through well-defined ports
- **Separation of Concerns**: Business logic isolated from infrastructure
- **Composability**: Reusable components that combine into complex workflows
- **Observability**: Built-in tracing, metrics, and event streaming

---

## 🏗️ Layered Architecture

### 📚 Layer Hierarchy Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    🎯 Configuration Layer                   │
│              (YAML Definitions & Schemas)                  │
├─────────────────────────────────────────────────────────────┤
│                   🚀 Application Layer                     │
│         (Use Cases: Orchestrator, AgentBuilder)            │
├─────────────────────────────────────────────────────────────┤
│                    🧠 Domain Layer                         │
│            (Core Business Logic: DAG, NodeSpec)            │
├─────────────────────────────────────────────────────────────┤
│                   🔌 Interface Layer                       │
│               (Ports: Abstract Protocols)                  │
├─────────────────────────────────────────────────────────────┤
│                 🔧 Infrastructure Layer                    │
│            (Adapters: Concrete Implementations)            │
├─────────────────────────────────────────────────────────────┤
│                   🌐 External Services                     │
│              (LLMs, Databases, APIs, Tools)                │
└─────────────────────────────────────────────────────────────┘
```

### Module Architecture

```
hexai/
├── core/
│   ├── domain/          # Core business logic (DAG, NodeSpec)
│   ├── application/     # Use cases (Orchestrator, AgentBuilder)
│   └── ports/           # Interface definitions (LLM, Database)
├── adapters/            # External service implementations
└── agent_factory/       # YAML-based agent workflow builders
```

---

## 🎯 Layer 1: Configuration Layer

### Purpose & Responsibility
The **declarative definition layer** where AI agent workflows are described in human-readable YAML format. This layer enables low-code development by translating business requirements into executable agent specifications.

### Key Components
- **Agent Workflow Files**: Declarative agent workflow definitions
- **Node Configurations**: Individual agent and processing step specifications
- **Dependency Declarations**: Execution order and data flow definitions
- **Parameter Schemas**: Input/output type definitions with Pydantic models

### Architectural Role
- **Business-IT Bridge**: Allows non-technical stakeholders to create AI workflows
- **Configuration Management**: Centralizes workflow definitions outside of code
- **Version Control**: Enables tracking and rollback of agent workflow changes
- **Environment Adaptation**: Different configurations for different deployment environments

### Design Principles
- **Declarative over Imperative**: Describe "what" not "how"
- **Self-Documenting**: YAML serves as both configuration and documentation
- **Technology Agnostic**: No implementation details, only business logic
- **Composable**: Agents can be mixed and matched across different workflows

### Example Configuration Structure
```yaml
# Standard agent workflow configuration format
name: research_analysis_workflow
description: Multi-agent research and analysis system

nodes:
  - type: function
    id: data_preparation
    depends_on: []

  - type: llm
    id: intelligent_analysis
    params:
      prompt_template: "Analyze: {{data_preparation.results}}"
      output_schema: {insights: str, confidence: float}
    depends_on: [data_preparation]

  - type: agent
    id: decision_synthesis
    params:
      max_steps: 3
      available_tools: [calculator, validator]
    depends_on: [intelligent_analysis]
```

---

## 🚀 Layer 2: Application Layer

### Purpose & Responsibility
The **use case orchestration layer** that translates business requirements into executable workflows. This layer contains the main application services that coordinate domain objects.

### Key Components

#### **Orchestrator** - Pure Execution Engine
- **Responsibility**: Pure DAG execution engine
- **Capabilities**: Wave-based parallel execution, dependency management, event emission
- **Design**: Stateless service that takes pre-built graphs and executes them
- **Isolation**: No knowledge of YAML parsing or node creation

#### **PipelineBuilder** - YAML to DAG Translation
- **Responsibility**: Configuration translation service
- **Capabilities**: YAML parsing, function registration, graph construction
- **Design**: Transforms declarative configurations into executable DirectedGraph objects
- **Validation**: Ensures configuration correctness before execution

#### **Context Manager** - Execution State Coordination
- **Responsibility**: Execution state coordination
- **Capabilities**: Cross-node memory, execution tracing, session management
- **Design**: Shared state object passed through entire execution chain
- **Lifecycle**: Created per pipeline execution, destroyed after completion

#### **Event System** - Observable Execution Monitoring
- **Responsibility**: Observable execution monitoring
- **Capabilities**: Real-time event emission, observer pattern implementation
- **Design**: Decoupled event generation and consumption
- **Extensibility**: Pluggable observers for different monitoring needs

### Architectural Role
- **Use Case Implementation**: Implements the main business use cases (build pipeline, execute pipeline)
- **Coordination Hub**: Orchestrates interactions between domain objects
- **External Interface**: Provides the main API surface for pipeline consumers
- **State Management**: Manages execution state and cross-cutting concerns

### Design Principles
- **Single Responsibility**: Each service has one clear purpose
- **Dependency Injection**: All external dependencies provided through constructor
- **Stateless Design**: Services don't maintain state between operations
- **Event-Driven**: Operations emit events for monitoring and integration

---

## 🧠 Layer 3: Domain Layer

### Purpose & Responsibility
The **pure business logic layer** containing core concepts and rules. This layer has zero external dependencies and represents the heart of the framework's business logic.

### Key Components

#### **DirectedGraph** - DAG Structure & Validation
- **Responsibility**: DAG structure and validation
- **Capabilities**: Node management, cycle detection, dependency analysis, wave calculation
- **Design**: Immutable value object representing a complete workflow
- **Intelligence**: Understands execution order, parallelization opportunities, and validation rules

#### **NodeSpec** - Individual Workflow Step Definition
- **Responsibility**: Individual workflow step definition
- **Capabilities**: Function reference, dependency declaration, parameter storage
- **Design**: Immutable value object with fluent API for construction
- **Composition**: Building block for complex workflows

#### **Wave Calculator** - Parallel Execution Planning
- **Responsibility**: Parallel execution planning
- **Capabilities**: Topological sorting, dependency analysis, parallelization optimization
- **Design**: Pure algorithm with no side effects
- **Output**: Execution waves where nodes in same wave can run concurrently

#### **Validation Engine** - Business Rule Enforcement
- **Responsibility**: Business rule enforcement
- **Capabilities**: Cycle detection, dependency validation, type compatibility checking
- **Design**: Fail-fast validation with detailed error messages
- **Coverage**: Structural validation, semantic validation, type safety

### Architectural Role
- **Business Logic Core**: Contains all core workflow concepts and rules
- **Framework Heart**: Other layers build upon these fundamental concepts
- **Pure Logic**: No external dependencies, completely testable in isolation
- **Domain Language**: Provides vocabulary for discussing workflow concepts

### Design Principles
- **Pure Functions**: No side effects, deterministic behavior
- **Immutability**: Objects cannot be modified after creation
- **Rich Domain Model**: Objects contain both data and behavior
- **Explicit Dependencies**: All relationships clearly declared

---

## 🔌 Layer 4: Interface Layer (Ports)

### Purpose & Responsibility
The **abstraction boundary layer** that defines contracts for external services without implementing them. This layer enables dependency inversion and testability.

### Key Components

#### **LLM Port** - Language Model Interaction Abstraction
- **Responsibility**: Language model interaction abstraction
- **Contract**: Message-based communication interface
- **Capabilities**: Async response generation, conversation handling
- **Abstraction**: Hides specific LLM implementation details

#### **Database Port** - Data Persistence Abstraction
- **Responsibility**: Data persistence abstraction
- **Contract**: Schema introspection and query interface
- **Capabilities**: Table schema access, relationship discovery, metadata retrieval
- **Abstraction**: Database-agnostic data access

#### **Tool Router Port** - External Tool Integration Abstraction
- **Responsibility**: External tool integration abstraction
- **Contract**: Tool discovery and execution interface
- **Capabilities**: Dynamic tool routing, parameter handling
- **Abstraction**: Tool-agnostic execution framework

#### **Memory Port** - Long-term Storage Abstraction
- **Responsibility**: Long-term storage abstraction
- **Contract**: Key-value storage interface
- **Capabilities**: Async get/set operations, persistence guarantees
- **Abstraction**: Storage-agnostic memory management

#### **Ontology Port** - Business Knowledge Abstraction
- **Responsibility**: Business knowledge abstraction
- **Contract**: Business concept and relationship access
- **Capabilities**: Entity discovery, relationship traversal, context retrieval
- **Abstraction**: Knowledge-source-agnostic business intelligence

### Architectural Role
- **Dependency Inversion**: Allows business logic to define what it needs, not how it's provided
- **Testing Enablement**: Provides clear interfaces for mock implementations
- **Technology Independence**: Business logic doesn't depend on specific technologies
- **Contract Definition**: Establishes clear expectations for external services

### Design Principles
- **Interface Segregation**: Small, focused interfaces rather than large ones
- **Stable Abstractions**: Interfaces change less frequently than implementations
- **Protocol Definition**: Clear contracts with explicit expectations
- **Minimal Surface Area**: Only expose what's absolutely necessary

---

## 🔧 Layer 5: Infrastructure Layer (Adapters)

### Purpose & Responsibility
The **implementation layer** that provides concrete implementations of the abstract interfaces defined in the Interface Layer. This layer handles all external service integration.

### Key Components

#### **Production Adapters**
- **LLM Adapters**: OpenAI, Anthropic, Local model integrations
- **Database Adapters**: PostgreSQL, MySQL, MongoDB implementations
- **Tool Adapters**: Web APIs, calculation engines, file systems
- **Memory Adapters**: Redis, database, file-based storage
- **Ontology Adapters**: Enterprise knowledge systems, graph databases

#### **Mock Adapters**
- **Testing Infrastructure**: Controllable, predictable implementations for testing
- **Development Support**: Fast, local implementations for development
- **Demonstration Tools**: Simple implementations for tutorials and examples
- **Integration Testing**: Reliable implementations for CI/CD pipelines

#### **Enhanced Adapters**
- **Composite Implementations**: Adapters that combine multiple services
- **Caching Layers**: Performance optimization adapters
- **Retry Logic**: Reliability-enhanced adapters
- **Monitoring Integration**: Observability-enhanced adapters

### Architectural Role
- **External Integration**: Bridges framework to external services and technologies
- **Technology Binding**: Implements technology-specific integration logic
- **Performance Optimization**: Handles caching, connection pooling, etc.
- **Error Handling**: Manages external service failures and retries

### Design Principles
- **Adapter Pattern**: Translates between framework interfaces and external service APIs
- **Fail-Safe Design**: Graceful handling of external service failures
- **Configuration Driven**: Externalized configuration for different environments
- **Resource Management**: Proper cleanup and resource lifecycle management

---

## 🌐 Layer 6: External Services

### Purpose & Responsibility
The **external systems layer** representing third-party services, databases, APIs, and tools that the framework integrates with. This layer is outside the framework's control.

### Service Categories

#### **AI Services**
- **Language Models**: GPT, Claude, Llama, etc.
- **Embedding Services**: OpenAI embeddings, local embedding models
- **Specialized AI**: Image generation, speech recognition, etc.

#### **Data Services**
- **Databases**: PostgreSQL, MySQL, MongoDB, etc.
- **Data Warehouses**: Snowflake, BigQuery, Redshift
- **Graph Databases**: Neo4j, ArangoDB
- **Search Engines**: Elasticsearch, Solr

#### **Business Services**
- **Knowledge Systems**: Enterprise ontologies, wikis, documentation
- **APIs**: REST services, GraphQL endpoints
- **File Systems**: Local storage, cloud storage (S3, GCS)
- **Message Queues**: RabbitMQ, Kafka, SQS

#### **Infrastructure Services**
- **Monitoring**: Prometheus, DataDog, New Relic
- **Logging**: ELK stack, Splunk, CloudWatch
- **Secret Management**: HashiCorp Vault, AWS Secrets Manager
- **Configuration**: Consul, etcd, environment variables

### Architectural Role
- **Service Provider**: Provides actual capabilities that the framework orchestrates
- **External Dependency**: Outside framework control, must be handled gracefully
- **Integration Point**: Where framework meets existing enterprise infrastructure
- **Capability Source**: Provides the actual AI, data, and business capabilities

---

## 🔄 Cross-Layer Interactions

### Data Flow Patterns

#### **Configuration to Execution Flow**
1. **Configuration Layer**: YAML defines business workflow
2. **Application Layer**: PipelineBuilder parses YAML into DirectedGraph
3. **Domain Layer**: DirectedGraph validates structure and calculates execution waves
4. **Application Layer**: Orchestrator executes graph using injected ports
5. **Interface Layer**: Ports define contracts for external service access
6. **Infrastructure Layer**: Adapters implement port contracts
7. **External Services**: Provide actual capabilities (LLM responses, data, etc.)

#### **Event Flow**
1. **Domain Layer**: Business events (node started, completed, failed)
2. **Application Layer**: Event system captures and routes events
3. **Infrastructure Layer**: Observers process events (logging, metrics, alerting)
4. **External Services**: Events may trigger external integrations

### Dependency Rules

#### **Inward Dependencies Only**
- **Configuration** depends on nothing
- **Application** depends on Domain and Interface layers
- **Domain** depends on nothing (pure business logic)
- **Interface** depends on standard library only
- **Infrastructure** depends on Interface layer and external SDKs
- **External Services** are independent

#### **Abstraction Levels**
- **High-Level Policy** (Domain): Business rules and workflow logic
- **Medium-Level Policy** (Application): Use case orchestration
- **Low-Level Detail** (Infrastructure): Technology-specific implementation

---

## 🔧 Node Type Architecture

### Factory Pattern for Node Creation

Each node type implements a factory pattern for consistent creation across the framework:

### Node Type Capabilities

| Node Type | Purpose | Key Features |
|-----------|---------|--------------|
| **FunctionNode** | Python function execution | Dependency injection, async support |
| **LLMNode** | Single LLM interactions | Template rendering, JSON parsing |
| **ReActAgentNode** | Multi-step AI reasoning | Tool calling, iterative thinking |
| **LoopNode** | Iterative control flow | Conditional termination, state tracking |
| **ConditionalNode** | Branching logic | Data-driven routing decisions |

### 🤖 Agent Nodes - Intelligent Autonomous Components

Agent nodes represent the most sophisticated components in the hexAI framework, combining LLM-powered reasoning with structured tool access to create intelligent, autonomous workflow participants.

#### **Core Capabilities**
- **Multi-step Reasoning**: Break down complex problems into manageable steps
- **Dynamic Tool Usage**: Autonomously decide which tools to use and when
- **Context Awareness**: Maintain conversation state across reasoning iterations
- **Adaptive Behavior**: Adjust strategy based on intermediate results
- **Error Recovery**: Handle tool failures and retry with alternative approaches

#### **Architecture Pattern**

```python
# Agent Factory Pattern
agent_factory = ReActAgentNode()

# Tool Integration
search_tool = ToolDefinition(
    name="search",
    simplified_description="Search for information",
    parameters=[ToolParameter(name="query", param_type="str", required=True)]
)

# Agent Creation
agent = agent_factory(
    name="research_agent",
    main_prompt="Research the following topic: {{input}}",
    available_tools=[search_tool],
    config=AgentConfig(max_steps=3, tool_call_style=ToolCallFormat.MIXED)
)
```

#### **Tool Router Integration**

The framework provides multiple tool router implementations for different use cases:

**MockToolRouter** - For rapid prototyping and testing:
```python
from hexai.adapters.mock.mock_tool_router import MockToolRouter

# String responses for quick mocking
mock_router = MockToolRouter()

# Custom async functions
async def custom_weather(location: str) -> str:
    return f"Weather in {location}: Sunny, 22°C"

mock_router.add_tool("weather", custom_weather)
```

**SimpleToolRouter** - For real async function execution:
```python
# See examples/10_agent_nodes.py for a complete implementation
# The SimpleToolRouter class demonstrates real async tool execution
real_router = SimpleToolRouter()  # Built-in tools: search, calculate, analyze
```

**Custom Router** - For advanced integration needs:
```python
class CustomToolRouter(ToolRouter):
    async def call_tool(self, tool_name: str, params: dict[str, Any]) -> Any:
        # Your custom tool execution logic
        pass
```

#### **Agent Configuration**

Agents support extensive configuration for different use cases:

```python
config = AgentConfig(
    max_steps=5,                           # Maximum reasoning iterations
    tool_call_style=ToolCallFormat.MIXED,  # Tool invocation format
    # Additional configuration options for timeout, retry, etc.
)
```

#### **Best Practices**

1. **Start with Mock Tools**: Use `MockToolRouter` for rapid development and testing
2. **Async Tool Functions**: Always use async functions for better performance
3. **Structured Results**: Return dictionaries from tools for easy downstream processing
4. **Error Handling**: Implement graceful error handling in tool functions
5. **Focused Tools**: Keep individual tools simple and single-purpose

#### **Monitoring & Debugging**

All tool routers provide call history for debugging:

```python
# Track tool usage
history = router.get_call_history()
for call in history:
    print(f"Tool: {call['tool_name']}, Result: {call['result']}")

# Monitor agent reasoning
result = await orchestrator.run(graph, input_data)
agent_result = result['agent_name']
print(f"Steps: {len(agent_result.reasoning_steps)}")
print(f"Tools used: {agent_result.tools_used}")
```

For comprehensive documentation and examples, see:
- [Agent Nodes README](app/application/nodes/README.md) - Detailed implementation guide
- [Example 10](../../../examples/10_agent_nodes.py) - Complete working example
- [Tool Router Port](app/ports/tool_router.py) - Interface specification

---

## 🎭 Multi-Agent Orchestration Patterns

### Sequential Coordination
**Use Case**: Document processing pipeline where each agent builds on previous work

**Architectural Pattern**: Linear dependency chain with progressive information enrichment

### Parallel Coordination
**Use Case**: Multi-perspective analysis requiring specialist viewpoints

**Architectural Pattern**: Parallel execution with coordinated synthesis

### Iterative Refinement
**Use Case**: Quality improvement through feedback loops

**Architectural Pattern**: Loop-based refinement with conditional termination

---

## 🔍 Event-Driven Observability

### Event Architecture

The framework generates comprehensive events throughout execution:

- **Pipeline Events**: Lifecycle events (started, completed, failed)
- **Node Events**: Individual node execution tracking
- **LLM Events**: Language model interaction monitoring
- **Tool Events**: External tool usage tracking

### Observer Pattern Implementation

Built-in observers handle different monitoring aspects:

- **LoggingObserver**: Structured logging for execution tracking
- **MetricsObserver**: Performance metrics collection
- **NodeObserver**: Detailed node-level execution tracking
- **WebSocketObserver**: Real-time event streaming
- **FileObserver**: Event persistence to disk

---

## 🧪 Testing Architecture

### Layered Testing Strategy

**Unit Tests**: Individual components in isolation
- Domain layer components tested without external dependencies
- Pure business logic validation
- Node factory behavior verification

**Integration Tests**: Component interaction
- Application layer service coordination
- Pipeline building and validation
- Event system behavior

**End-to-End Tests**: Complete workflow execution
- Full pipeline execution with mock adapters
- Production-like scenarios with controlled inputs
- Performance and scalability validation

### Mock Adapter Ecosystem

Comprehensive mock implementations enable thorough testing:
- **MockLLM**: Configurable LLM responses
- **MockToolRouter**: Tool call simulation
- **MockDatabaseAdapter**: Database schema mocking
- **MockOntologyPort**: Business ontology simulation


---

---

The hexAI framework provides a **production-ready foundation** for building sophisticated AI workflows that integrate seamlessly with enterprise infrastructure. Through its layered architecture, clean separation of concerns, and comprehensive observability, hexAI enables organizations to build **maintainable, testable, and scalable AI operations** that deliver consistent business value.

## What is hexAI?

hexAI is a **lightweight orchestration framework** that provides the minimal building blocks for creating and executing DAG-based workflows. It focuses on doing one thing well: orchestrating the execution of nodes in a directed graph.

### Core Features
- 🎯 **DAG Orchestration**: Define and execute computational graphs
- 🔧 **Node System**: Simple, extensible node types (function, LLM, agent)
- ✅ **Validation Framework**: Pluggable validation with strategies (strict/coerce/passthrough)
- 🪝 **Event Hooks**: Observe and react to execution events
- 🚀 **Async Execution**: Built-in support for concurrent node execution

### What hexAI is NOT
- ❌ **Not a data pipeline platform** (see enterprise pipelines layer)
- ❌ **Not a distributed computing framework** (single-process execution)
- ❌ **Not an MLOps platform** (no model management, monitoring, etc.)
- ❌ **Not a workflow UI** (code-first, no visual editor)
- ❌ **Not opinionated about storage** (bring your own persistence)

## Quick Start

```python
from hexai import DirectedGraph, NodeSpec, Orchestrator

# Define your functions
async def fetch_data(input_data: str) -> dict:
    return {"query": input_data, "results": [1, 2, 3]}

async def process_data(input_data: dict) -> str:
    total = sum(input_data["results"])
    return f"Processed {len(input_data['results'])} items, total: {total}"

# Build the graph
graph = DirectedGraph()
graph.add(NodeSpec("fetch", fetch_data))
graph.add(NodeSpec("process", process_data).after("fetch"))

# Execute
orchestrator = Orchestrator()
results = await orchestrator.run(graph, "my query")
print(results["process"])  # "Processed 3 items, total: 6"
```

## Installation

```bash
pip install hexai  # Coming soon
```

## Core Concepts

### 1. Nodes
Nodes are the basic units of computation:

```python
# Function node - wrap any Python function
NodeSpec("my_node", my_function)

# LLM node - for language model interactions
from hexai.nodes import llm_node
llm_node("analyzer", "Analyze this: {{input}}")

# Agent node - for complex reasoning
from hexai.nodes import agent_node
agent_node("researcher", tools=[search_tool, calculate_tool])
```

### 2. Graphs
Graphs define the execution flow:

```python
graph = DirectedGraph()
graph.add(NodeSpec("a", func_a))
graph.add(NodeSpec("b", func_b).after("a"))
graph.add(NodeSpec("c", func_c).after("a", "b"))
```

### 3. Validation
Flexible validation strategies:

```python
from hexai.validation import strict_validator, coerce_validator

# Strict: fail on type mismatch
orchestrator = Orchestrator(validator=strict_validator())

# Coerce: attempt type conversion
orchestrator = Orchestrator(validator=coerce_validator())
```

### 4. Events
Hook into the execution lifecycle:

```python
from hexai.events import EventObserver

class MyObserver(EventObserver):
    async def on_node_completed(self, event):
        print(f"Node {event.node_name} completed in {event.execution_time}s")

orchestrator = Orchestrator()
orchestrator.add_observer(MyObserver())
```

## Architecture

hexAI follows a clean, layered architecture:

```
hexai/
├── domain/          # Core domain models (NodeSpec, DirectedGraph)
├── application/     # Application services (Orchestrator, Nodes)
├── validation/      # Validation framework
└── events/         # Event system
```

## 📚 Examples & Learning Path

hexAI comes with **20 comprehensive examples** that provide a progressive learning path from basic concepts to advanced patterns. All examples are located in the `examples/` directory and can be run individually or as a complete learning journey.

See `README` there for more.


## Extending hexAI

### Custom Nodes
Create your own node types:

```python
from hexai.nodes import BaseNodeFactory

class MyCustomNode(BaseNodeFactory):
    def __call__(self, name: str, **kwargs):
        def custom_logic(input_data, **ports):
            # Your logic here
            return result

        return NodeSpec(name, custom_logic, **kwargs)
```

### Custom Validators
Add domain-specific validation:

```python
from hexai.validation import TypeConverter, register_converter

class DateConverter(TypeConverter):
    def can_convert(self, source_type, target_type):
        return source_type == str and target_type == datetime

    def convert(self, value, target_type):
        return datetime.fromisoformat(value)

register_converter(DateConverter())
```

### Custom Ports
Extend with your own ports:

```python
# Coming soon with Port Registry System
@register_port("vector_store")
class VectorStorePort:
    async def search(self, query: str) -> list:
        # Your implementation
        pass
```

## Best Practices

1. **Keep nodes focused**: Each node should do one thing well
2. **Use type hints**: Enable validation and better IDE support
3. **Handle errors gracefully**: Use validation strategies appropriately
4. **Avoid side effects**: Nodes should be deterministic when possible
5. **Test your graphs**: Use the provided testing utilities

## Performance

hexAI is designed for performance:

- **Minimal overhead**: < 1ms per node orchestration
- **Concurrent execution**: Automatic parallelization of independent nodes
- **Zero dependencies**: Only requires Pydantic
- **Memory efficient**: Streaming execution for large graphs

## Comparison with Other Tools

| Feature | hexAI | Airflow | Prefect | Dagster |
|---------|-------|---------|---------|---------|
| Focus | Simple DAG execution | Enterprise workflows | Data pipelines | Data assets |
| Weight | Lightweight (~10k LOC) | Heavy | Medium | Heavy |
| Dependencies | Minimal (Pydantic) | Many | Many | Many |
| Learning Curve | Low | High | Medium | High |
| Use Case | Embedded orchestration | Scheduled jobs | Data engineering | ML pipelines |

## When to Use hexAI

✅ **Use hexAI when you need:**
- Simple DAG orchestration in your application
- Lightweight embedded workflow engine
- Clean, testable pipeline code
- Minimal external dependencies

❌ **Don't use hexAI when you need:**
- Distributed execution across machines
- Complex scheduling (cron, calendar)
- Built-in UI for monitoring
- Heavy enterprise features

For enterprise features, see the [Pipelines](../pipelines) layer built on top of hexAI.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.
