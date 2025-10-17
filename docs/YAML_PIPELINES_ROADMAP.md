# YAML Pipelines ROADMAP v2.0

**YAML-based Agent Workflow Builder with Multi-Agent Coordination Patterns**

---

## Phase 1: Registry System Foundation

**Objective:** Build unified registry architecture with official/user extension distinction

### 1.1 Core Registry System (`hexdag.core.registry`)

* **NodeTypeRegistry:** Framework node factories with build-time verification
* **Build script:** Auto-generate official nodes list from `hexdag.core.application.nodes`
* **Official verification:** Simple module path comparison for security
* **Extension API:** Clean registration interface for user nodes

### 1.2 YAML Pipelines Registry System (`hexdag.pipeline_builder.registry`)

* **FunctionRegistry:** Per-pipeline callable function management
* **PipelineRegistry:** Enhanced PipelineCatalog with namespacing/versioning
* **MacroRegistry:** Template discovery and expansion system
* **CustomNodeRegistry:** User-defined node types specific to agent\_factory

### 1.3 Registry Integration

* **Cross-registry resolution:** Nodes, functions, pipelines, macros
* **Metadata system:** Tags, descriptions, versioning, official status
* **Discovery API:** Unified interface for finding components

---

## Phase 2: Enhanced YAML Builder with Registry Integration

**Objective:** K8s-style declarative manifests with full registry support

### 2.1 Registry-Aware `YamlPipelineBuilder`

* **Node resolution:** Use NodeTypeRegistry for official + custom nodes
* **Function binding:** Use FunctionRegistry for pipeline functions
* **Cross-references:** Resolve pipeline/macro imports via registries
* **Validation:** Schema validation using registry metadata

### 2.2 Multi-Document YAML Support

```yaml
apiVersion: hexdag/v1
kind: Node
metadata:
  name: custom-consensus
  namespace: my-org
spec:
  type: CustomConsensusNode  # From pipeline_builder custom registry
---
apiVersion: hexdag/v1
kind: Pipeline
spec:
  imports:
    - pipeline: "security/content-filter:1.0.0"
  nodes: [custom-consensus]
```

### 2.3 Cross-Reference Resolution

* **Pipeline dependencies:** `security/content-filter:1.0.0#validator-node`
* **Node imports:** Reference nodes from other pipelines/namespaces
* **Macro expansion:** Template imports with parameter injection

---

## Phase 3: Custom Extension System in YAML Pipelines

**Objective:** YAML Pipelines as extension platform for custom components

### 3.1 Custom Node Framework (`pipeline_builder/custom/`)

* **CustomNodeBase:** Base class for agent\_factory-specific nodes
* **CoordinationNodes:** Multi-agent coordination patterns
* **SecurityNodes:** Specialized validation/filtering nodes
* **DataNodes:** Enhanced data processing capabilities
* **Auto-registration:** Custom nodes auto-discovered and registered

### 3.2 Custom Pipeline Patterns (`pipeline_builder/patterns/`)

* **Multi-agent templates:** Consensus, hierarchical, feedback loops
* **Security patterns:** Content filtering, prompt injection detection
* **Data patterns:** ETL, semantic search, knowledge graphs
* **Coordination patterns:** Event-driven, reactive, distributed

### 3.3 Custom Function Library (`pipeline_builder/functions/`)

* **Validation functions:** Schema validation, content safety
* **Coordination functions:** Voting, consensus, delegation
* **Data functions:** Parsing, transformation, aggregation
* **Integration functions:** API calls, database operations

---

## Phase 4: Dual Registry System (Standard + MLflow)

**Objective:** Development registry + production lifecycle management

### 4.1 Standard Registry (`pipeline_builder/registry/standard.py`)

* **Development focus:** Fast iteration, simple discovery
* **Auto-registration:** Pipelines, macros, custom components
* **Metadata:** Basic tags, descriptions, relationships
* **Execution:** Direct workflow execution

### 4.2 MLflow-Style Registry (`pipeline_builder/registry/versioned.py`)

* **Production focus:** Lifecycle management, governance
* **Versioning:** Semantic versioning for all components
* **Stages:** Development → Staging → Production → Archived
* **Experiments:** Performance tracking, A/B testing
* **Artifacts:** Configuration snapshots, validation results

### 4.3 Registry Promotion Pipeline

* **Development → Staging:** Automated testing, validation
* **Staging → Production:** Manual approval, performance gates
* **Rollback:** Version-based rollback capabilities
* **Analytics:** Usage metrics, performance monitoring

---

## Phase 5: Macro System with Custom Injection

**Objective:** Reusable templates with custom component injection

### 5.1 Macro Architecture (`pipeline_builder/macros/`)

* **MacroDefinition:** Inherits pipeline patterns, adds templating
* **MacroParam:** Custom injection points for nodes/functions/policies
* **MacroExpander:** Template → concrete YAML using PromptTemplate
* **Custom injection:** User nodes, functions, validation policies

### 5.2 Essential Macro Collection

#### Multi-Agent Coordination (`macros/coordination/`)

* **Chain-of-Thought:** `{{reasoning_agent}} + {{validator_agent}}`
* **Consensus Network:** `{{voter_agents}} + {{consensus_function}}`
* **Manager-Worker:** `{{manager_agent}} + {{worker_pool}}`
* **Feedback Loop:** `{{executor}} + {{critic}} + {{improver}}`

#### Security & Validation (`macros/security/`)

* **Prompt Injection Detection:** `{{classifier}} + {{rules_engine}}`
* **Content Safety:** `{{safety_agents}} + {{escalation_policy}}`
* **Authorization:** `{{auth_function}} + {{access_validator}}`
* **Audit Trail:** `{{logger}} + {{compliance_checker}}`

#### Data Processing (`macros/data/`)

* **ETL Pipeline:** `{{extractor}} + {{transformer}} + {{loader}}`
* **Document Analysis:** `{{parser}} + {{analyzer}} + {{summarizer}}`
* **Semantic Search:** `{{embedder}} + {{retriever}} + {{ranker}}`

---

# Phase 6: Governance Framework and Flexible Orchestration Patterns

**Objective:** Governance framework and flexible orchestration patterns

### 6.1 Policy Framework (`pipeline_builder/policies/`)

* **PolicyDefinition:** Base class for governance rules
* **Security Policies:** RBAC, data classification, access control
* **Resource Policies:** Rate limiting, quota management, SLA enforcement
* **Validation Policies:** Schema enforcement, content filtering, compliance
* **Custom Policies:** User-defined governance and business rules
* **Policy Registry:** Discovery and management of policy components
* **Policy Enforcement:** Integration with pipeline execution

### 6.2 Multi-Orchestrator Architecture (`pipeline_builder/orchestration/`)

* **Orchestrator Configurations:** Multiple orchestrator setups per deployment
* **Resource Optimization:** CPU, memory, and throughput-optimized configs
* **Workload Distribution:** Route pipelines to appropriate orchestrators
* **Configuration Management:** YAML-based orchestrator definitions
* **Dynamic Scaling:** Auto-scaling orchestrator pools based on load
* **Health Monitoring:** Orchestrator availability and performance tracking

### 6.3 Policy-Orchestrator Integration

* **Policy-aware routing:** Route workflows based on policy requirements
* **Compliance orchestrators:** Dedicated orchestrators for regulated workloads
* **Resource governance:** Policy-based resource allocation and limits
* **Audit integration:** Policy enforcement tracking and compliance reporting

### 6.4 Advanced Coordination with Policies

* **Policy-driven workflows:** Macros that adapt based on active policies
* **Conditional orchestration:** Route to different orchestrators based on policies
* **Multi-tenant isolation:** Policy-based workspace and resource separation
* **Governance automation:** Self-healing and policy violation remediation

---

## Phase 7: Observability and Event Sink

**Objective:** Unified observability and event-driven extensibility

### 7.1 Observability Refactor

* **Registry metrics:** Usage patterns, performance tracking
* **Custom component monitoring:** User extension health checks
* **Distributed tracing:** Cross-component execution flows
* **Dashboard integration:** Grafana-compatible metrics
* **Unified observability API:** Standardized metrics and logging hooks

### 7.2 Event Sink (`pipeline_builder/events/`)

* **EventDefinition:** Standard schema for registry + orchestration events
* **EventRouter:** Routes events to sinks (logging, monitoring, external systems)
* **Pluggable Sinks:** Kafka, Webhooks, CloudWatch, Prometheus exporters
* **Event Correlation:** Link registry, policy, and orchestration events
* **Replay & Auditing:** Store events for compliance and debugging
* **Integration with Policies:** Policy violations emitted as structured events
