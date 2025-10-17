# hexDAG YAML Pipelines: Architectural Design and Multi-Tier Framework

**Enterprise AI Workflow Orchestration Through Principled Architecture**

---

## Executive Summary

hexDAG YAML Pipelines addresses the fundamental challenges in AI orchestration with a **three-tier architecture**.
Unlike retrofitted frameworks that couple business logic, orchestration, and cloud operations, hexDAG separates these concerns into:

1. **Core Engine** → mathematical primitives and execution runtime
2. **YAML Pipelines** → declarative workflows, registries, macros, and policies
3. **Plug-and-Play Integrations** → provider-specific extensions for enterprise adoption

This architecture ensures **testability, extensibility, and governance** while maintaining developer ergonomics and enterprise robustness.

---

## 🏗️ Three-Tier Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Tier 3: Plug-and-Play Integrations (hexdag[provider])        │
│ • Native cloud/data integrations (Azure, AWS, GCP, Spark)    │
│ • Cloud monitoring, IAM, compliance orchestrators            │
│ • Kubernetes-native deployment (CRDs, autoscaling)           │
├─────────────────────────────────────────────────────────────┤
│ Tier 2: YAML Pipelines (hexdag.pipeline_builder)                  │
│ • YAML pipelines & macros                                    │
│ • Dual Registry System (standard + versioned)                │
│ • Policy & governance framework                              │
│ • Multi-orchestrator routing and workload specialization     │
│ • Official vs unofficial nodes, cross-registry resolution    │
├─────────────────────────────────────────────────────────────┤
│ Tier 1: Core Engine (hexdag.core)                             │
│ • DAG orchestration & execution primitives                   │
│ • Async-first runtime                                        │
│ • Pydantic type safety and validation                        │
│ • Event-driven observability                                 │
│ • Port/adapter abstraction for external dependencies         │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Tier 1: Core Engine

### Design Philosophy: Pure Computational Model

The Core is **domain-agnostic**. It provides the mathematical foundation for orchestrating workflows, independent of AI, cloud, or business context.

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

## 🏗️ Tier 2: YAML Pipelines

### Design Philosophy: Systematic Composition & Customization

YAML Pipelines transforms the Core into an AI orchestration platform, balancing **low-code workflow design** with **deep extensibility** through registries.

---

#### **1. Registry System (Customization Backbone)**

The registry architecture is what makes hexDAG **extensible without modifying the framework**.

* **NodeTypeRegistry** → official and custom nodes
* **FunctionRegistry** → user-defined pipeline functions
* **PipelineRegistry** → discoverable workflows, namespace + versioned
* **MacroRegistry** → reusable template definitions
* **PolicyRegistry** → governance components

**Official vs Unofficial Nodes**

* Official: auto-verified at build-time from `hexdag.core.application.nodes`
* Unofficial: user-registered through clean APIs
* Metadata (`official: true/false`, tags, version) ensures traceability and governance

**Dual Registry Modes**

* **Standard (Development)** → lightweight, auto-discovery, hot-reload
* **Versioned (MLflow-style)** → semantic versioning, staging → prod promotion, rollback, experiment tracking

**Cross-Registry Resolution**

* Pipelines can reference nodes, macros, policies across registries:
  `security/content-filter:1.0.0#validator-node`

---

#### **2. YAML Pipelines (K8s-Style)**

* Declarative manifests for nodes, pipelines, macros.
* Multi-document YAML supported.
* Schema validation powered by registry metadata.

```yaml
apiVersion: hexdag/v1
kind: Pipeline
spec:
  imports:
    - pipeline: "security/content-filter:1.0.0"
  nodes:
    - id: analyzer
      type: AgentNode
      params: { prompt: "Analyze: {{input}}" }
```

---

#### **3. Macro System**

* Parameterized templates with injection points.
* Eliminates copy-paste workflows.
* Macro collections:

  * **Multi-agent coordination** (consensus, manager-worker, feedback loops)
  * **Security/compliance** (prompt injection detection, audit trail)
  * **Data processing** (ETL, semantic search, document analysis)

---

#### **4. Policy & Governance Engine**

* **Security**: RBAC, classification, access control
* **Resource**: rate limits, quotas, SLAs
* **Validation**: schema enforcement, content filtering
* **Custom**: org-specific rules

Policies integrate at:

* Pre-execution validation
* Node-level execution
* Post-execution auditing
* Cross-workflow governance

---

#### **5. Multi-Orchestrator Support**

* Specialized orchestrators (CPU, memory, throughput, compliance).
* Workload routing driven by policies and registry metadata.
* Enables isolation of regulated workloads.

---

### YAML Pipelines Benefits

* **Customization** → Registries provide clean extension points
* **Low-code orchestration** → YAML pipelines for rapid design
* **Reusable patterns** → Macros standardize workflows
* **Governed execution** → Policies and registry metadata enforce compliance
* **Scalable execution** → Multi-orchestrator specialization

---

## 🌐 Tier 3: Plug-and-Play Integrations

### Design Philosophy: Native, Modular, Vendor-Neutral

While Core and YAML Pipelines remain cloud-agnostic, enterprises need **native integration** with their platforms. Tier 3 provides that via modular plug-ins.

---

#### **Supported Extensions**

* **hexdag\[azure]** → Azure AD, Key Vault, Service Bus, Monitor, Managed Identity
* **hexdag\[aws]** → IAM roles, Lambda, CloudWatch, SQS/SNS, Secrets Manager
* **hexdag\[gcp]** → Vertex AI, Pub/Sub, Cloud Monitoring, BigQuery
* **hexdag\[spark]** → Distributed DAG execution, Delta Lake, MLlib, streaming
* **hexdag\[k8s]** → Kubernetes-native CRDs, autoscaling, service mesh integration

---

#### **Design Principles**

1. **Native Integration**

   * Use provider IAM, secrets, monitoring directly (not lowest-common-denominator abstractions).

2. **Backward Compatibility**

   * Pipelines run with or without extensions.
   * Extensions enhance, never break Core/Factory.

3. **Multi-Cloud Flexibility**

   * Same YAML pipeline deploys across providers.
   * Event sinks (Kafka, Pub/Sub, CloudWatch) unify observability.
   * Compliance orchestrators integrate with provider-native audit trails.

---

### Plug-and-Play Benefits

* Enterprise-grade **security & compliance** (IAM, audit, policies).
* **Observability** with cloud-native tools (Azure Monitor, CloudWatch, Prometheus).
* **Scalability** across Spark, Kubernetes, and serverless runtimes.
* **No vendor lock-in**: workflows remain portable.

---

## 📚 Conclusion: Architecture as Advantage

hexDAG’s **three-tier architecture** provides:

* **Tier 1 (Core)** → Deterministic async DAG engine with type safety and observability
* **Tier 2 (YAML Pipelines)** → Customizable orchestration with registries, YAML, macros, policies, and multi-orchestrator routing
* **Tier 3 (Integrations)** → Plug-and-play provider extensions for enterprise adoption

By **separating computation, composition, and integration**, hexDAG delivers both:

* **Developer velocity** → clean APIs, low-code pipelines, reusable macros
* **Enterprise rigor** → governance, observability, versioning, compliance
* **Future-proofing** → modular integrations and systematic extensibility

hexDAG is architecture-first orchestration — designed for **scalability, maintainability, and enterprise adoption**.
