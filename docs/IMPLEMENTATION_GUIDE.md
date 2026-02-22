# 🚀 hexDAG Implementation Guide

> **Build production-ready AI agent workflows using the hexDAG framework**

## 📖 Table of Contents

1. [Framework Overview](#framework-overview)
2. [Architecture Principles](#architecture-principles)
3. [Core Components](#core-components)
4. [Node Development Guidelines](#node-development-guidelines)
5. [Port & Adapter Pattern](#port--adapter-pattern)
6. [Context System](#context-system)
7. [Event System & Observability](#event-system--observability)
8. [Testing Methodologies](#testing-methodologies)
9. [Error Handling Best Practices](#error-handling-best-practices)
10. [Production Deployment](#production-deployment)

---

## 🏗️ Framework Overview

### hexDAG Architecture

hexDAG is the core orchestration framework that enables building sophisticated AI agent workflows through declarative configuration. It transforms complex AI workflows into **deterministic, testable, and maintainable** systems by implementing the six pillars:

1. **🔄 Async-First Architecture**: Non-blocking execution for maximum performance
2. **📊 Event-Driven Observability**: Real-time monitoring of agent actions
3. **✅ Pydantic Validation Everywhere**: Type safety at every layer
4. **🏗️ Hexagonal Architecture**: Clean separation of business logic and infrastructure
5. **📝 Composable Declarative Files**: Build complex workflows from simple components
6. **🔀 DAG-Based Orchestration**: Intelligent dependency management and parallelization

### Layered Architecture

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
hexdag/
├── core/
│   ├── domain/          # Core business logic (DAG, NodeSpec)
│   ├── application/     # Use cases (Orchestrator, AgentBuilder)
│   └── ports/           # Interface definitions (LLM, Database)
├── adapters/            # External service implementations
└── pipeline_builder/       # YAML-based agent workflow builders
```

hexDAG implements hexagonal architecture specifically designed for AI agent orchestration and data science workflows. It provides separation between business logic, external dependencies, and execution concerns through well-defined layers.




## 🏛️ Architecture Principles

### 1. Hexagonal Architecture

hexDAG's hexagonal architecture creates clean boundaries between AI agent logic and external services, enabling true portability and testability.

**Core Domain** (`hexdag.core.domain`):
The domain layer contains pure business logic for agent orchestration, DAG management, and workflow execution. It remains independent of any external AI services or infrastructure concerns.

**Application Layer** (`hexdag.builtin`):
This layer orchestrates agent workflows and manages execution flow. It handles event streaming, context management, and coordination between agents while maintaining separation from infrastructure.

**Ports** (`hexdag.kernel.ports`):
Ports define abstract interfaces for external services like LLMs, vector databases, and tool providers. They establish contracts that ensure loose coupling between the framework and external dependencies.

**Essential Port Example:**
```python
from typing import Any, Protocol

class LLM(Protocol):
    """Language model interface for AI agents."""

    async def aresponse(self, messages: list[dict[str, Any]]) -> str | None:
        """Generate response from messages."""
        ...

    async def astream(self, messages: list[dict[str, Any]]) -> Any:
        """Stream response from messages."""
        ...
```

**Adapters** (`hexdag.adapters`):
Adapters implement port interfaces to connect with actual AI services. They handle API translation, error handling, and service-specific concerns.

**Essential Adapter Example:**
```python
class OpenAIAdapter(LLM):
    """Adapter for OpenAI models."""

    def __init__(self, model: Any) -> None:
        self.model = model

    async def aresponse(self, messages: list[dict[str, Any]]) -> str | None:
        """Generate response using factory model."""
        try:
            response = await self.model.aresponse(messages)
            return response.content
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return None
```

### 2. Deterministic Execution

The framework ensures that workflow execution is predictable and reproducible. Given the same inputs and configuration, a workflow will always follow the same execution path and produce the same results.

**DAG-Based Workflows:**
Workflows are represented as Directed Acyclic Graphs (DAGs), where nodes represent processing steps and edges represent dependencies.

**Essential DAG Example:**
```python
from hexdag import DirectedGraph, NodeSpec

# Create a directed acyclic graph
graph = DirectedGraph()

# Add nodes with dependencies
node_a = NodeSpec("processor", processor_function)
node_b = NodeSpec("analyzer", analyzer_function, depends_on=["processor"])
node_c = NodeSpec("validator", validator_function, depends_on=["analyzer"])

graph.add(node_a)
graph.add(node_b)
graph.add(node_c)
```

**Wave-Based Execution:**
The orchestrator computes execution waves through topological sorting, allowing nodes to execute in parallel when possible while respecting dependencies.

**Context Management:**
The context system provides shared state across nodes while maintaining isolation. Each node can read from and write to the context, but the framework ensures that state changes are predictable and traceable.

### 3. Type Safety

The framework emphasizes strong typing throughout to catch errors at development time rather than runtime.

**Pydantic Integration:**
The framework integrates deeply with Pydantic for automatic validation, schema inference, and type safety.

**Essential Type Safety Example:**
```python
from pydantic import BaseModel, Field
from typing import Any

class MyFunctionInput(BaseModel):
    """Input model with validation and documentation."""
    text: str = Field(..., description="Text to process")
    priority: int = Field(default=1, ge=1, le=10, description="Priority level")

class MyFunctionOutput(BaseModel):
    """Output model with clear structure."""
    processed_text: str = Field(..., description="Processed text result")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Processing confidence")

async def my_function(
    input_data: MyFunctionInput,
    context: Context,
    **ports: Any
) -> MyFunctionOutput:
    """Process text with full type safety."""
    context.add_trace("my_function", f"Processing {len(input_data.text)} characters")

    # Pydantic handles all input validation automatically
    processed = await process_text(input_data.text, ports.get("llm"))

    return MyFunctionOutput(
        processed_text=processed["text"],
        confidence=processed["confidence"],
        original_input=input_data
    )
```

---

## 🔧 Core Components

### 1. DirectedGraph

The DirectedGraph is the core data structure that represents a workflow. It manages nodes, their dependencies, and the execution order.

**Key Responsibilities:**
- Managing node specifications and their relationships
- Computing execution waves through topological sorting
- Validating graph structure and dependencies
- Ensuring schema compatibility between connected nodes

**Essential Graph Operations:**
```python
from hexdag.core.domain.dag import DirectedGraph, NodeSpec

# Create and populate graph
graph = DirectedGraph()

# Add nodes
graph.add(NodeSpec("input", input_function))
graph.add(NodeSpec("process", process_function, depends_on=["input"]))
graph.add(NodeSpec("output", output_function, depends_on=["process"]))

# Validate graph
graph.validate()

# Get execution waves
waves = graph.waves()
print(f"Execution waves: {waves}")
# Output: [['input'], ['process'], ['output']]
```

**Graph Properties:**
- **Acyclic**: No circular dependencies allowed (though one can use the loop node)
- **Directed**: Clear dependency relationships
- **Validated**: Automatic validation of structure and schemas
- **Composable**: Graphs can be combined and reused

### 2. NodeSpec

NodeSpec represents a single processing step in a workflow. It's an immutable specification that defines what a node does, what it expects as input, what it produces as output, and what other nodes it depends on.

**Essential NodeSpec Example:**
```python
from hexdag.core.domain.dag import NodeSpec

# Create node specification
node_spec = NodeSpec(
    name="text_processor",
    fn=process_text_function,
    in_type=TextInput,  # Pydantic model
    out_type=TextOutput,  # Pydantic model
    deps={"input_validator"},  # Dependencies
    params={"max_length": 1000}  # Configuration
)

# Add dependencies fluently
node_spec = node_spec.after("input_validator", "data_loader")
```

**Specification Elements:**
- **Name**: Unique identifier within the graph
- **Function**: The actual processing logic to execute
- **Input/Output Types**: Type specifications for validation
- **Dependencies**: Other nodes that must complete first
- **Metadata**: Additional configuration parameters

**Immutability:**
NodeSpecs are immutable to ensure that workflow definitions are predictable and thread-safe.

### 3. Orchestrator

The Orchestrator is the execution engine that runs workflows. It manages the execution of nodes, handles concurrency, manages resources, and coordinates the overall workflow execution.

**Essential Orchestrator Usage:**
```python
from hexdag import Orchestrator, Context

# Create orchestrator
orchestrator = Orchestrator(max_concurrent_nodes=5)

# Execute workflow
context = Context()
ports = {"llm": llm_adapter, "database": db_adapter}
input_data = {"query": "test query"}

results = await orchestrator.run(
    graph=my_graph,
    initial_input=input_data,
    context=context,
    ports=ports
)

print(f"Execution results: {results}")
```

**Execution Strategy:**
- **Wave-Based**: Executes nodes in dependency order
- **Concurrent**: Runs independent nodes in parallel
- **Resource-Limited**: Controls concurrency to prevent resource exhaustion
- **Error-Handling**: Graceful handling of node failures

**Key Features:**
- **Deterministic**: Same inputs always produce same execution path
- **Observable**: Comprehensive event system for monitoring
- **Resilient**: Handles failures gracefully without affecting other nodes
- **Scalable**: Configurable concurrency limits

---

## 🎯 Node Development Guidelines

### 1. Node Types

The framework supports several types of nodes, each optimized for different use cases:

**Function Nodes:**
Function nodes execute arbitrary Python functions. They're the most flexible node type and can handle any processing logic.

**Essential Function Node Example:**
```python
async def my_function(
    input_data: Any,
    context: Context,
    **ports: Any
) -> dict[str, Any]:
    """Function node implementation."""

    # Add execution trace
    context.add_trace("my_function", "Starting processing")

    # Validate required ports
    if "llm" not in ports:
        raise ValueError("LLM port is required")

    # Process data
    result = await process_data(input_data, ports["llm"])

    # Store in memory for other nodes
    context.set_memory("my_function_result", result)

    # Return with original data preserved
    return {"processed_result": result, **input_data}
```

**LLM Nodes:**
LLM nodes are specialized for language model interactions. They handle prompt templating, response parsing, and LLM-specific concerns.

**Agent Nodes:**
Agent nodes implement multi-step reasoning with tool usage. They can make multiple calls to external tools and maintain conversation state across steps.

**Loop Nodes:**
Loop nodes implement iterative processing with conditions. They can repeat processing steps until certain conditions are met.

**Conditional Nodes:**
Conditional nodes implement branching logic based on data values. They can route execution down different paths based on conditions.

### 2. Node Development Principles

**Single Responsibility:**
Each node should have a single, well-defined responsibility. This makes nodes easier to test, debug, and reuse.

**Stateless Design:**
Nodes should be stateless when possible, relying on the context for shared state. This makes nodes more predictable and easier to test.

**Error Handling:**
Nodes should handle errors gracefully and provide meaningful error messages. They should not crash the entire workflow when possible.

**Type Safety:**
Nodes should use strong typing for inputs and outputs. This helps catch errors early and provides better documentation.

**Observability:**
Nodes should provide meaningful traces and use the context system for logging and debugging information.

### 3. Pydantic Integration

The framework deeply integrates with Pydantic for automatic validation and schema inference.

**Essential Pydantic Pattern:**
```python
from pydantic import BaseModel, Field
from typing import Any

class MyFunctionInput(BaseModel):
    """Input model with validation and documentation."""
    text: str = Field(..., description="Text to process")
    priority: int = Field(default=1, ge=1, le=10, description="Priority level")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class MyFunctionOutput(BaseModel):
    """Output model with clear structure."""
    processed_text: str = Field(..., description="Processed text result")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Processing confidence")
    features: dict[str, Any] = Field(..., description="Extracted features")
    original_input: MyFunctionInput = Field(..., description="Original input for traceability")

async def my_function(
    input_data: MyFunctionInput,
    context: Context,
    **ports: Any
) -> MyFunctionOutput:
    """Process text with full type safety and automatic schema inference."""
    context.add_trace("my_function", f"Processing {len(input_data.text)} characters")

    # Pydantic handles all input validation automatically
    processed = await process_text(input_data.text, ports.get("llm"))

    # Return typed model instance
    return MyFunctionOutput(
        processed_text=processed["text"],
        confidence=processed["confidence"],
        features=processed["features"],
        original_input=input_data
    )
```

---

## 🔌 Port & Adapter Pattern

### 1. Port Design

Ports define abstract interfaces that external services must implement. They provide clear contracts that ensure loose coupling between the framework and external dependencies.

**Essential Port Examples:**
```python
from typing import Any, Protocol

class LLM(Protocol):
    """Language model interface."""

    async def aresponse(self, messages: list[dict[str, Any]]) -> str | None:
        """Generate response from messages."""
        ...

    async def astream(self, messages: list[dict[str, Any]]) -> Any:
        """Stream response from messages."""
        ...

class Database(Protocol):
    """Database interface."""

    async def get_schema(self, table_name: str) -> dict[str, Any]:
        """Get table schema."""
        ...

    async def execute_query(self, query: str) -> list[dict[str, Any]]:
        """Execute SQL query."""
        ...

class Memory(Protocol):
    """Long-term memory interface."""

    async def store(self, key: str, value: Any) -> None:
        """Store value with key."""
        ...

    async def retrieve(self, key: str) -> Any | None:
        """Retrieve value by key."""
        ...
```

**Interface Design:**
Ports are designed as Python protocols, providing clear contracts without imposing implementation details. This allows for flexible implementations while maintaining type safety.

**Service Categories:**
The framework defines ports for common AI workflow needs:
- **LLM Port**: Language model interactions
- **Database Port**: Data storage and retrieval
- **Memory Port**: Long-term memory storage
- **Tool Router Port**: External tool integration
- **Ontology Port**: Business knowledge access
- **Embedding Selector Port**: Vector similarity search

### 2. Adapter Implementation

Adapters implement port interfaces and handle the actual integration with external services. They translate between the framework's expectations and the external service's API.

**Essential Adapter Examples:**
```python
class LLMFactoryAdapter(LLM):
    """Adapter for LLM factory models."""

    def __init__(self, model: Any) -> None:
        self.model = model

    async def aresponse(self, messages: list[dict[str, Any]]) -> str | None:
        """Generate response using factory model."""
        try:
            response = await self.model.aresponse(messages)
            return response.content
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return None

class MockLLM(LLM):
    """Mock LLM for testing."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.call_count = 0

    async def aresponse(self, messages: list[dict[str, Any]]) -> str | None:
        """Return predefined responses."""
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return "Mock response"
```

**Implementation Patterns:**
- **Factory Adapters**: Handle different model types and configurations
- **Enhanced Adapters**: Provide additional functionality beyond basic ports
- **Mock Adapters**: Provide test implementations for development and testing
- **Production Adapters**: Handle real external services with proper error handling

**Error Handling:**
Adapters should handle external service failures gracefully and provide meaningful error messages. They should not crash the framework when external services are unavailable.

### 3. Port Registration

Port implementations are registered and used throughout the workflow execution.

**Essential Port Usage:**
```python
# Create port implementations
ports = {
    "llm": LLMFactoryAdapter(production_llm_model),
    "database": EnhancedDatabaseAdapter(production_db),
    "memory": ProductionMemoryAdapter(redis_connection),
    "ontology": ProductionOntologyAdapter(ontology_service),
}

# Use in orchestrator
results = await orchestrator.run(graph, input_data, context, ports)
```

---


## 📊 Event System & Observability

### 1. Event Types

The framework provides a comprehensive event system for monitoring and debugging workflow execution.

**Available Event Types:**
```python
from hexdag.builtin.events.base import EventType

# Pipeline events
EventType.PIPELINE_STARTED
EventType.PIPELINE_COMPLETED
EventType.PIPELINE_FAILED

# Node events
EventType.NODE_STARTED
EventType.NODE_COMPLETED
EventType.NODE_FAILED

# Wave events
EventType.WAVE_STARTED
EventType.WAVE_COMPLETED

# LLM events
EventType.LLM_PROMPT_GENERATED
EventType.LLM_RESPONSE_RECEIVED

# Tool events
EventType.TOOL_CALLED
EventType.TOOL_COMPLETED

# Validation events
EventType.VALIDATION_WARNING
```

### 2. Event Observers

The event system uses the observer pattern to provide flexible monitoring capabilities.

**Essential Observer Setup:**
```python
from hexdag.builtin.events.observers import (
    LoggingObserver,
    MetricsObserver,
    FileObserver
)

# Set up comprehensive monitoring
event_manager = ObserverManager()
event_manager.subscribe(LoggingObserver())
event_manager.subscribe(MetricsObserver())
event_manager.subscribe(FileObserver("pipeline_events.log"))

# Use in orchestrator
orchestrator = Orchestrator(event_manager=event_manager)
```

**Built-in Observers:**
- **Logging Observer**: Output events to logging systems
- **Metrics Observer**: Collect performance and usage metrics
- **File Observer**: Write events to files for analysis
- **WebSocket Observer**: Stream events to real-time dashboards

**Custom Observers:**
Developers can implement custom observers for specific monitoring needs. Observers can filter events, aggregate data, or integrate with external monitoring systems.

### 3. Production Monitoring

The event system provides comprehensive monitoring capabilities for production environments.

**Essential Monitoring Setup:**
```python
from hexdag.builtin.events import ObserverManager
from hexdag.builtin.events.observers import LoggingObserver, MetricsObserver
import logging

# Production monitoring
event_manager = ObserverManager()
event_manager.subscribe(LoggingObserver(log_level=logging.INFO))
event_manager.subscribe(MetricsObserver())

# Use in orchestrator
orchestrator = Orchestrator(event_manager=event_manager)
```

**Performance Monitoring:**
- **Execution Times**: Track how long each node takes to execute
- **Resource Usage**: Monitor memory and CPU usage
- **Throughput**: Track requests per second and concurrent executions
- **Error Rates**: Monitor failure rates and error patterns

**Business Monitoring:**
- **User Activity**: Track user interactions and usage patterns
- **Feature Usage**: Monitor which features are being used
- **Success Rates**: Track successful vs failed workflow executions
- **Custom Metrics**: Track business-specific metrics

---

## 🧪 Testing Methodologies

### 1. Unit Testing

Unit testing focuses on testing individual components in isolation, ensuring they work correctly without external dependencies.

**Essential Unit Test Pattern:**
```python
import pytest
from unittest.mock import AsyncMock
from hexdag import MockLLM, MockDatabaseAdapter

class TestMyFunction:
    """Test my function implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.ports = {
            "llm": MockLLM(["test response"]),
            "database": MockDatabaseAdapter(),
        }

    @pytest.mark.asyncio
    async def test_function_success(self):
        """Test successful execution."""
        input_data = {"query": "test query"}

        result = await my_function(input_data, self.context, **self.ports)

        assert "processed_result" in result
        assert result["query"] == "test query"
        assert len(self.context.trace) > 0

    @pytest.mark.asyncio
    async def test_function_missing_port(self):
        """Test error handling."""
        input_data = {"query": "test"}
        ports_without_llm = {"database": MockDatabaseAdapter()}

        with pytest.raises(ValueError, match="LLM port is required"):
            await my_function(input_data, self.context, **ports_without_llm)
```

**Testing Patterns:**
- **Mock Dependencies**: Use mock adapters for external services
- **Edge Cases**: Test boundary conditions and error scenarios
- **Performance**: Test with realistic data volumes
- **Isolation**: Ensure tests don't interfere with each other

### 2. Integration Testing

Integration testing focuses on testing how components work together, ensuring the overall system functions correctly.

**Essential Integration Test Pattern:**
```python
import pytest
from hexdag import DirectedGraph, NodeSpec, Orchestrator

@pytest.mark.asyncio
async def test_workflow_execution():
    """Test complete workflow execution."""
    # Create graph
    graph = DirectedGraph()

    # Add nodes
    node_a = NodeSpec("processor", processor_function)
    node_b = NodeSpec("analyzer", analyzer_function, depends_on=["processor"])

    graph.add(node_a)
    graph.add(node_b)

    # Create ports
    ports = {
        "llm": MockLLM(["analysis result"]),
        "database": MockDatabaseAdapter(),
    }

    # Execute
    orchestrator = Orchestrator()
    context = Context()
    input_data = {"query": "test input"}

    results = await orchestrator.run(graph, input_data, context, ports)

    assert "analyzer" in results
    assert results["analyzer"]["status"] == "success"
```

**Testing Patterns:**
- **End-to-End**: Test complete workflows from start to finish
- **Realistic Data**: Use realistic data volumes and patterns
- **Error Scenarios**: Test how the system handles various failure modes
- **Performance**: Measure and optimize system performance

### 3. Mock Adapters

The framework provides comprehensive mock implementations for all ports, making testing reliable and predictable.

**Essential Mock Usage:**
```python
from hexdag.stdlib.adapters.mock import MockLLM, MockDatabaseAdapter
from hexdag.kernel.ports.tool_router import ToolRouter

# Use in tests -- configure mock responses from MockLLM.
ports = {
    "llm": MockLLM(["response1", "response2"]),
    "database": MockDatabaseAdapter(),
    "tool_router": ToolRouter(tools={
        "search": lambda query="", **kw: {"results": []},
    }),
}
```

**Mock Features:**
- **Predictable Responses**: Configure mock adapters to return specific responses
- **Error Simulation**: Simulate various error conditions and failure modes
- **Performance Simulation**: Simulate realistic response times and resource usage
- **State Tracking**: Track how mock adapters are used during testing

---

## ⚠️ Error Handling Best Practices

### 1. Input Validation

Robust input validation is essential for preventing errors and ensuring system reliability.

**Essential Validation Pattern:**
```python
from pydantic import BaseModel, ValidationError

class MyFunctionInput(BaseModel):
    """Input validation model."""
    query: str
    user_id: str
    priority: int = 1

async def robust_function(input_data: Any, context: Context, **ports: Any) -> dict[str, Any]:
    """Function with comprehensive error handling."""

    try:
        # Validate input using Pydantic
        validated_input = MyFunctionInput(**input_data)
        context.add_trace("function", "Input validation successful")

    except ValidationError as e:
        context.add_trace("function", f"Input validation failed: {e}")
        raise ValueError(f"Invalid input: {e}")

    # Process validated data
    result = await process_data(validated_input, ports)

    return {"result": result, **validated_input.model_dump()}
```

**Validation Strategies:**
- **Schema Validation**: Use Pydantic models for automatic validation
- **Business Rules**: Validate business-specific constraints
- **Sanitization**: Clean and sanitize input data
- **Type Checking**: Ensure data types match expectations

### 2. Port Validation

Port validation ensures that external services are available and functioning correctly.

**Essential Port Validation:**
```python
def validate_ports(**ports: Any) -> None:
    """Validate required ports."""
    required = ["llm", "database"]
    missing = [p for p in required if p not in ports or ports[p] is None]

    if missing:
        raise ValueError(f"Missing required ports: {missing}")

    # Validate port interfaces
    if not hasattr(ports["llm"], "aresponse"):
        raise ValueError("LLM port must implement 'aresponse' method")

    if not hasattr(ports["database"], "execute_query"):
        raise ValueError("Database port must implement 'execute_query' method")
```

**Validation Strategies:**
- **Health Checks**: Verify that external services are healthy
- **Interface Validation**: Ensure ports implement required interfaces
- **Capability Checks**: Verify that services have required capabilities
- **Error Handling**: Test how services handle various error conditions

### 3. Graceful Degradation

The system should handle failures gracefully without affecting overall functionality.

**Essential Degradation Pattern:**
```python
async def resilient_function(input_data: Any, context: Context, **ports: Any) -> dict[str, Any]:
    """Function with fallback strategies."""

    try:
        # Primary processing
        result = await primary_processing(input_data, ports)
        context.add_trace("function", "Primary processing successful")
        return {"result": result, "method": "primary", **input_data}

    except Exception as primary_error:
        context.add_trace("function", f"Primary failed: {primary_error}")

        try:
            # Fallback processing
            result = await fallback_processing(input_data, ports)
            context.add_trace("function", "Fallback successful")
            return {"result": result, "method": "fallback", **input_data}

        except Exception as fallback_error:
            # Return safe result to prevent workflow failure
            context.add_trace("function", f"All methods failed")
            return {
                "result": None,
                "method": "failed",
                "error": str(fallback_error),
                **input_data
            }
```

**Degradation Strategies:**
- **Fallback Logic**: Provide alternative processing paths when primary paths fail
- **Partial Results**: Return partial results when complete processing isn't possible
- **Error Recovery**: Automatically retry failed operations when appropriate
- **User Communication**: Clearly communicate what happened and what users can expect



---

## 🎭 Multi-Agent Orchestration Patterns

### Sequential Agent Chain Pattern

**Use Case**: Linear workflows where each agent builds on previous agent's work

**Architecture**: Progressive Information Enhancement
```
Research Agent → Analysis Agent → Decision Agent → Implementation Agent
```

**YAML Configuration Example**:
```yaml
name: sequential_multiagent_workflow
description: Sequential chain of specialized agents

nodes:
  - type: agent
    id: research_agent
    params:
      initial_prompt_template: |
        You are a Research Agent. Your goal: {{goal}}
        Research thoroughly and provide structured findings.
      max_steps: 4
      available_tools: ["web_search", "database_lookup"]
    depends_on: []

  - type: agent
    id: analysis_agent
    params:
      initial_prompt_template: |
        Analyze research findings: {{research_agent.research_findings}}
        Generate insights and recommendations.
      max_steps: 3
    depends_on: [research_agent]

  - type: agent
    id: decision_agent
    params:
      initial_prompt_template: |
        Based on analysis: {{analysis_agent.insights}}
        Make final recommendations and implementation plan.
      max_steps: 2
    depends_on: [analysis_agent]
```

### Parallel Coordination Pattern

**Use Case**: Complex problems requiring multiple specialist perspectives

**Architecture**: Specialist Analysis with Coordinator Synthesis
```
Technical Specialist ──┐
                       ├── Coordination Agent → Final Decision
Business Specialist ───┘
```

**YAML Configuration Example**:
```yaml
name: parallel_multiagent_workflow
description: Parallel specialist agents with coordinator synthesis

nodes:
  # Parallel Specialist Agents (Wave 1)
  - type: agent
    id: technical_specialist
    params:
      initial_prompt_template: |
        You are a Technical Specialist Agent.
        Analyze from technical perspective: {{task_description}}
      max_steps: 5
      available_tools: ["tech_database", "architecture_analyzer"]
    depends_on: []

  - type: agent
    id: business_specialist
    params:
      initial_prompt_template: |
        You are a Business Specialist Agent.
        Analyze from business perspective: {{task_description}}
      max_steps: 5
      available_tools: ["market_research", "financial_calculator"]
    depends_on: []

  # Coordination Agent (Wave 2)
  - type: agent
    id: coordination_agent
    params:
      initial_prompt_template: |
        Synthesize findings from specialists:
        Technical: {{technical_specialist.technical_assessment}}
        Business: {{business_specialist.business_assessment}}
        Create unified strategy balancing both perspectives.
      max_steps: 6
    depends_on: [technical_specialist, business_specialist]
```

### Hierarchical Loop-Based Pattern

**Use Case**: Iterative refinement and complex multi-round negotiations

**Architecture**: Supervisor-Managed Iterative Improvement
```
Supervisor Agent ──→ Loop Controller ──→ Worker Agents ──→ Validator ──→ Loop Decision
```

**YAML Configuration Example**:
```yaml
name: hierarchical_multiagent_workflow
description: Hierarchical agents with loops for iterative improvement

nodes:
  - type: agent
    id: supervisor_agent
    params:
      initial_prompt_template: |
        You are the Supervisor Agent managing specialist agents.
        Define tasks and quality criteria for: {{primary_goal}}
      max_steps: 3
    depends_on: []

  - type: loop
    id: agent_iteration_loop
    params:
      max_iterations: 3
      success_condition: |
        lambda data: (
          data.get('validation_result', {}).get('quality_score', 0) >= 8 and
          data.get('validation_result', {}).get('meets_criteria', False)
        )
    depends_on: [supervisor_agent]

  - type: agent
    id: worker_agent
    params:
      initial_prompt_template: |
        You are a Worker Agent in iteration {{agent_iteration}}.
        Task: {{supervisor_agent.task_assignments}}
        Previous feedback: {{validation_feedback}}
        Improve based on feedback.
      max_steps: 4
    depends_on: [agent_iteration_loop]

  - type: agent
    id: validator_agent
    params:
      initial_prompt_template: |
        Evaluate work quality: {{worker_agent.analysis_report}}
        Score 1-10 and provide improvement feedback.
      max_steps: 2
    depends_on: [worker_agent]
```

### YAML Pipeline Development

The YAML pipelines enables **declarative agent workflow development** through YAML configuration:

**Key Features:**
- **Auto-Discovery**: Automatic detection of pipeline implementations
- **Schema Introspection**: Dynamic type discovery for UI generation
- **Compilation**: Pre-optimized execution for production
- **Multi-Agent Coordination**: Built-in patterns for agent collaboration

**Development Workflow:**
1. Define agents and dependencies in YAML
2. Implement business logic functions
3. Register functions with YAML pipelines
4. Compile for production deployment

For detailed YAML pipeline implementation, see the [YAML Pipelines Implementation Guide](../hexdag/pipeline_builder/PIPELINES_IMPLEMENTATION_GUIDE.md).

---

*This guide provides a comprehensive overview of the hexAI framework architecture and implementation patterns. Follow these principles to ensure consistency with the existing codebase and maintain high-quality, maintainable code.*
