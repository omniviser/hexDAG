# hexDAG YAML Pipelines: Architecture and Current Implementation

**Declarative AI Workflow Orchestration with Clean Separation of Concerns**

---

## Executive Summary

hexDAG YAML Pipelines provides a **declarative approach to AI workflow orchestration** built on a clean architectural foundation:

1. **Core Engine** (`hexdag.core`) → DAG orchestration, execution runtime, type safety
2. **YAML Pipeline Builder** (`hexdag.core.pipeline_builder`) → Declarative manifests with plugin architecture
3. **Registry System** (`hexdag.core.registry`) → Component discovery and namespace management

**Current Status:** Core engine and YAML builder are production-ready. Advanced features (policy framework, multi-orchestrator routing, cloud integrations) are planned for future releases.

This document describes the **current implementation** and architectural patterns. For planned features, see [YAML_PIPELINES_ROADMAP.md](YAML_PIPELINES_ROADMAP.md).

---

## 🏗️ Current Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: Registry System (hexdag.core.registry)              │
│ ✅ Component registration and discovery                      │
│ ✅ Namespace management (core, plugin, custom)               │
│ ✅ Type-based resolution (nodes, macros, adapters, tools)    │
│ ✅ Decorator-based component definition                      │
├─────────────────────────────────────────────────────────────┤
│ Layer 2: YAML Pipeline Builder (hexdag.core.pipeline_builder)│
│ ✅ K8s-style manifest format (apiVersion, kind, metadata)    │
│ ✅ Plugin architecture (preprocessing + entity plugins)      │
│ ✅ Environment variable resolution (${VAR})                  │
│ ✅ Jinja2 templating support                                 │
│ ✅ Macro system with expansion                               │
│ ✅ Multi-document YAML support                               │
├─────────────────────────────────────────────────────────────┤
│ Layer 1: Core Engine (hexdag.core)                           │
│ ✅ DAG orchestration & execution primitives                  │
│ ✅ Async-first runtime                                       │
│ ✅ Pydantic type safety and validation                       │
│ ✅ Event-driven observability                                │
│ ✅ Port/adapter abstraction for external dependencies        │
└─────────────────────────────────────────────────────────────┘

Legend: ✅ = Implemented | 🚧 = In Progress | 📋 = Planned
```

---

## ⚙️ Layer 1: Core Engine

### Design Philosophy: Pure Computational Model

The Core Engine is **domain-agnostic** and provides the mathematical foundation for orchestrating workflows, independent of AI, cloud, or business context.

#### **1. DAG as the Primary Abstraction**

* Provides dependency management and parallel execution.
* Alternatives (state machines, Petri nets, streams) rejected for rigidity or complexity.
* DAGs offer the right balance of **clarity, determinism, and concurrency**.

```python
graph = DirectedGraph()
graph.add(node_a)
graph.add(node_b, deps=["node_a"])
waves = graph.waves()   # Parallel execution plan
```

#### **2. Async-First Execution Model**

* Designed for I/O-bound AI workloads (LLM calls, APIs).
* Supports thousands of concurrent agents.
* Leverages Python’s `async/await` for scalability.

```python
async def execute_node(node: NodeSpec, inputs: dict):
    result = await node.fn(**inputs)
    return {"output": result}
```

#### **3. Type Safety with Pydantic**

* Every node specifies input/output schemas.
* IDE autocomplete, runtime validation, and living documentation.

```python
class Input(BaseModel):
    query: str
class Output(BaseModel):
    result: str
```

#### **4. Event-Driven Observability**

* Events over logs → structured, queryable, and decoupled.
* Covers pipeline, wave, node, and tool-level actions.

```python
@dataclass
class NodeExecutionStarted:
    node_id: str
    timestamp: datetime
```

#### **5. Port/Adapter Pattern (Hexagonal Architecture)**

* **Ports** define abstract interfaces (LLM, DB, ToolRouter, Ontology, Memory, Embeddings).
* **Adapters** implement them (OpenAIAdapter, RedisMemoryAdapter, SparkSQLAdapter).

**Benefits:**

* Testable with mocks
* Replaceable without modifying business logic
* Vendor-agnostic at the core

#### **6. Execution Context & Error Isolation**

* Context system propagates shared state with full traceability.
* Failures isolated at node-level; workflows degrade gracefully.
* Loop and conditional nodes supported as primitives for expressive orchestration.

---

## 🏗️ Layer 2: YAML Pipeline Builder

### Design Philosophy: Declarative Workflows with Plugin Architecture

The YAML Pipeline Builder transforms the Core Engine into a declarative orchestration platform through a clean plugin architecture.

---

### **Current Features**

#### **1. K8s-Style Manifest Format**

hexDAG uses a familiar Kubernetes-style declarative format:

```yaml
apiVersion: v1
kind: Pipeline
metadata:
  name: my-pipeline
  description: Pipeline description
spec:
  nodes:
    - kind: llm_node
      metadata:
        name: analyzer
      spec:
        prompt_template: "Analyze: {{input}}"
        model: gpt-4
        temperature: 0.7
        dependencies: []
```

**Key Fields:**
- `apiVersion` - Schema version (currently `v1`)
- `kind` - Resource type (`Pipeline`, `macro_invocation`)
- `metadata` - Name, description, annotations
- `spec` - Pipeline specification (nodes, ports, policies)

#### **2. Plugin Architecture**

The builder uses two plugin types:

**Preprocessing Plugins** (run before graph building):
- `EnvironmentVariablePlugin` - Resolves `${VAR}` and `${VAR:default}`
- `TemplatePlugin` - Jinja2 templating with context

**Entity Plugins** (build specific components):
- `MacroEntityPlugin` - Expands `macro_invocation` into subgraphs
- `NodeEntityPlugin` - Builds nodes from registry

This allows easy extension without modifying core builder logic.

#### **3. Environment Variable Resolution**

```yaml
spec:
  nodes:
    - kind: llm_node
      metadata:
        name: llm
      spec:
        model: ${MODEL_NAME:gpt-4}  # Falls back to gpt-4
        temperature: ${TEMPERATURE}   # Required if no default
```

Features:
- `${VAR}` - Required variable (error if missing)
- `${VAR:default}` - Optional with default value
- Type coercion - Automatic conversion to int, float, bool
- Recursive resolution - Works in nested dicts/lists

#### **4. Jinja2 Templating**

The TemplatePlugin provides Jinja2 templating with intelligent context-aware rendering:

**Build-time Templating** (YAML-level configuration):
```yaml
spec:
  variables:
    model_name: gpt-4
    node_prefix: analyzer
  nodes:
    - kind: llm_node
      metadata:
        name: "{{ spec.variables.node_prefix }}_llm"  # Rendered at build time
      spec:
        model: "{{ spec.variables.model_name }}"      # Rendered at build time
        template: "Analyze: {{input}}"                 # Preserved for runtime
```

**Runtime Templating** (node execution):
```yaml
spec:
  nodes:
    - kind: prompt_node
      metadata:
        name: prepare_data
      spec:
        template: "Data: {{input}}"  # Rendered at runtime with actual input

    - kind: llm_node
      metadata:
        name: analyzer
      spec:
        # This template is preserved and rendered at runtime with dependency outputs
        template: "Analyze this: {{prepare_data.text}}"
        dependencies: [prepare_data]
```

**Template Rendering Strategy:**

- **Metadata fields**: Rendered at build time using YAML config context
  - Example: `name: "{{ spec.variables.node_name }}"` → Resolves to static node name

- **Node spec fields**: Preserved for runtime rendering with execution context
  - Example: `template: "{{dependency.output}}"` → Rendered when node executes
  - Allows templates to access outputs from dependency nodes

- **Pipeline outputs**: Preserved for runtime rendering after execution
  - Example: `outputs: {result: "{{analyzer.analysis}}"}` → Rendered after nodes complete
  - Allows mapping node results to pipeline outputs

- **Pipeline spec fields**: Rendered at build time (except node specs and outputs)
  - Example: `variables`, `ports`, `policies` support templating

This two-phase approach enables:
- ✅ Dynamic pipeline configuration from environment/config at build time
- ✅ Dynamic prompt generation from dependency outputs at runtime
- ✅ Dynamic output mapping from node execution results
- ✅ Clean separation between configuration and execution concerns

#### **5. Macro System**

Macros enable reusable pipeline templates:

```yaml
# Define macro instance
- kind: macro_invocation
  metadata:
    name: my_rag_instance
  spec:
    macro: "rag_pipeline"  # macro name or full module path
    config:
      chunk_size: 512
      top_k: 5
    inputs:
      query: "{{user_query}}"
    dependencies: [previous_node]
```

Macros expand into DirectedGraph subgraphs at build time.

#### **6. Multi-Document YAML**

```yaml
---
apiVersion: v1
kind: Pipeline
metadata:
  name: dev-pipeline
spec:
  nodes: [...]
---
apiVersion: v1
kind: Pipeline
metadata:
  name: prod-pipeline
spec:
  nodes: [...]
```

Select environment: `builder.build_from_yaml_string(yaml, environment="prod")`

#### **7. Node Type Resolution**

Nodes are resolved using the module path resolver:

```yaml
- kind: llm_node                              # Built-in alias
- kind: hexdag.builtin.nodes.LLMNode          # Full module path
- kind: mypackage.nodes.CustomNode            # Custom node
```

#### **8. Function Node with Module Path Strings**

Function nodes now support declarative function references via module path strings:

```yaml
# Use any Python function without imports
- kind: function_node
  metadata:
    name: json_parser
  spec:
    fn: "json.loads"  # Module path string - resolved at build time
  dependencies: []

# Standard library functions
- kind: function_node
  metadata:
    name: base64_encoder
  spec:
    fn: "base64.b64encode"
  dependencies: []

# Your custom business logic
- kind: function_node
  metadata:
    name: order_processor
  spec:
    fn: "myapp.business.process_order"
    input_schema:
      order_id: str
      customer_id: str
    output_schema:
      status: str
      total: float
  dependencies: [validate_order]

# Third-party packages
- kind: function_node
  metadata:
    name: data_loader
  spec:
    fn: "pandas.read_csv"
  dependencies: []
```

**Benefits:**
- **Fully Declarative** - No Python imports needed in pipeline definitions
- **Git-Friendly** - Pure YAML with no code dependencies
- **Clear Errors** - Descriptive messages for invalid paths at build time
- **Universal** - Works with any Python function (stdlib, packages, custom code)

**Error Handling:**
```yaml
# Invalid module → "Could not import module from function path"
fn: "nonexistent.module.func"

# Invalid function → "Function 'bad_func' not found in module 'json'"
fn: "json.bad_func"

# Non-callable → "Resolved 'json.__version__' is not callable"
fn: "json.__version__"
```

---

### **YAML Pipeline Benefits**

✅ **Version Control** - Track workflows in Git like infrastructure code
✅ **Declarative** - What to do, not how to do it
✅ **Type Safe** - Schema validation before execution
✅ **Portable** - Same YAML across environments
✅ **Extensible** - Plugin architecture for custom behavior
✅ **Composable** - Macros for reusable patterns
✅ **Function Module Paths** - Reference any Python function via string paths

---

## 🔧 Layer 3: Registry System

### Design Philosophy: Component Discovery and Resolution

Components are resolved by their full Python module path using the resolver system.

#### **Current Features**

**1. Component Types:**
- Nodes (extend `BaseNodeFactory`)
- Macros (extend `ConfigurableMacro`)
- Adapters (implement port interfaces like `LLM`, `Memory`, `Database`)
- Tools (plain functions with type hints)

**2. Component Resolution:**

```python
from hexdag.core.resolver import resolve

# Resolve components by module path
CustomProcessor = resolve("myapp.nodes.CustomProcessor")
MyAdapter = resolve("myapp.adapters.MyAdapter")
```

**3. Using in YAML:**

```yaml
# Reference components by full module path
nodes:
  - kind: myapp.nodes.CustomProcessor
    metadata:
      name: processor1

# Built-in aliases available for convenience
nodes:
  - kind: llm_node  # Alias for hexdag.builtin.nodes.LLMNode
```

**4. Built-in Aliases:**
- `llm_node` → `hexdag.builtin.nodes.LLMNode`
- `function_node` → `hexdag.builtin.nodes.FunctionNode`
- `agent_node` → `hexdag.builtin.nodes.AgentNode`

---

## 📋 Planned Features

The following features are documented in roadmap but not yet implemented:

**Policy & Governance Framework:**
- PolicyRegistry for governance rules
- Pre/post execution policy enforcement
- RBAC, rate limiting, compliance policies

**Multi-Orchestrator Support:**
- Specialized orchestrators (CPU, memory, compliance)
- Policy-driven workload routing
- Dynamic scaling and health monitoring

**Dual Registry System:**
- Standard registry (current implementation)
- MLflow-style versioned registry (planned)
- Semantic versioning and lifecycle management
- Staging → production promotion

**Cloud Integrations:**
- `hexdag[azure]`, `hexdag[aws]`, `hexdag[gcp]`
- Native IAM, secrets, monitoring integration
- Kubernetes CRDs and operators

See [YAML_PIPELINES_ROADMAP.md](YAML_PIPELINES_ROADMAP.md) for complete roadmap.

---

## 📚 Conclusion

hexDAG's **layered architecture** delivers:

* **Layer 1 (Core)** → Deterministic async DAG engine with type safety
* **Layer 2 (YAML Builder)** → Declarative manifests with plugin architecture
* **Layer 3 (Registry)** → Component discovery and namespace management

By **separating orchestration, declaration, and discovery**, hexDAG provides:

* **Developer Velocity** → Clean APIs and declarative YAML
* **Extensibility** → Plugin architecture for customization
* **Type Safety** → Pydantic validation throughout
* **Testability** → Mock adapters and validation before execution

hexDAG is architecture-first orchestration — designed for **scalability, maintainability, and production use**.
