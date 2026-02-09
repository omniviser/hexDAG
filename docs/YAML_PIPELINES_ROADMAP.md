# YAML Pipelines ROADMAP

**Evolution of Declarative AI Workflow Orchestration**

---

## Status Legend

* ✅ **Implemented** - Available in current release
* 🚧 **In Progress** - Under active development
* 📋 **Planned** - Future roadmap item

---

## Phase 1: Registry System Foundation ✅

**Status:** ✅ **IMPLEMENTED**

**Objective:** Build unified registry architecture for component discovery

### 1.1 Component Resolution System (`hexdag.core.resolver`) ✅

* ✅ **Module Path Resolver:** Resolve components by full Python module path
* ✅ **Built-in Aliases:** Short names for built-in components (e.g., `llm_node`)
* ✅ **User Aliases:** Register custom short names for your components
* ✅ **Runtime Components:** Support for dynamically created components

### 1.2 Component Types Supported ✅

* ✅ **Node Factories:** Build NodeSpec instances from declarative config
* ✅ **Macro Definitions:** Reusable pipeline templates with expansion
* ✅ **Adapters:** Port implementations (LLM, database, memory, etc.)
* ✅ **Tools:** Agent tool functions with automatic schema generation
* ✅ **Policies:** Governance and validation components (basic support)

### 1.3 Resolver Features ✅

* ✅ **Module path resolution:** Import components by their Python path
* ✅ **Auto-discovery:** Built-in aliases registered at startup
* ✅ **Lazy loading:** Components imported on-demand
* ✅ **Error handling:** Clear error messages for missing components

---

## Phase 2: YAML Pipeline Builder ✅

**Status:** ✅ **IMPLEMENTED**

**Objective:** K8s-style declarative manifests with plugin architecture

### 2.1 Core Builder Features ✅

* ✅ **K8s-style manifests:** `apiVersion`, `kind`, `metadata`, `spec` format
* ✅ **Schema validation:** YamlValidator ensures correct structure
* ✅ **Plugin architecture:** Preprocessing and entity plugins
* ✅ **Multi-document YAML:** Environment-specific configurations
* ✅ **Registry integration:** Node resolution via registry
* ✅ **Error reporting:** Clear validation and build errors

### 2.2 Preprocessing Plugins ✅

* ✅ **EnvironmentVariablePlugin:** `${VAR}` and `${VAR:default}` resolution
* ✅ **TemplatePlugin:** Jinja2 templating in YAML values
* ✅ **Type coercion:** Automatic int, float, bool conversion
* ✅ **Recursive processing:** Nested dicts and lists

### 2.3 Entity Plugins ✅

* ✅ **MacroEntityPlugin:** Expand `macro_invocation` into subgraphs
* ✅ **NodeEntityPlugin:** Build all node types from YAML
* ✅ **Module path resolution:** Support full Python module paths
* ✅ **Dependency handling:** Explicit dependencies via `dependencies` field

### 2.4 YAML Manifest Format ✅

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
        dependencies: []

    - kind: macro_invocation
      metadata:
        name: rag
      spec:
        macro: "core:rag_pipeline"
        config:
          chunk_size: 512
        inputs:
          query: "{{user_query}}"
        dependencies: [analyzer]
```

---

## Phase 3: Advanced Registry Features 📋

**Status:** 📋 **PLANNED**

**Objective:** Enhanced registry with versioning and governance

### 3.1 Dual Registry System 📋

* 📋 **Standard Registry:** Current lightweight implementation
* 📋 **Versioned Registry:** MLflow-style lifecycle management
* 📋 **Semantic versioning:** Component versioning with semver
* 📋 **Stage management:** Development → Staging → Production
* 📋 **Rollback support:** Version-based rollback capabilities

### 3.2 Cross-Registry Resolution 📋

* 📋 **Versioned references:** `security/content-filter:1.0.0#validator-node`
* 📋 **Pipeline dependencies:** Import nodes from other pipelines
* 📋 **Macro libraries:** Shared macro collections
* 📋 **Version constraints:** Compatible version resolution

---

## Phase 4: Macro Library Expansion 📋

**Status:** 📋 **PLANNED**

**Objective:** Rich library of reusable macro templates

### 4.1 Multi-Agent Coordination Macros 📋

* 📋 **Chain-of-Thought:** Reasoning + validation pattern
* 📋 **Consensus Network:** Multi-agent voting and agreement
* 📋 **Manager-Worker:** Hierarchical task delegation
* 📋 **Feedback Loop:** Executor + critic + improver cycle

### 4.2 Security & Validation Macros 📋

* 📋 **Prompt Injection Detection:** Classifier + rules engine
* 📋 **Content Safety:** Safety filters with escalation
* 📋 **Authorization:** Auth validation and access control
* 📋 **Audit Trail:** Logging and compliance checking

### 4.3 Data Processing Macros 📋

* 📋 **ETL Pipeline:** Extract, transform, load pattern
* 📋 **Document Analysis:** Parse, analyze, summarize
* 📋 **Semantic Search:** Embed, retrieve, rank workflow
* 📋 **RAG Pipeline:** Retrieval-augmented generation (partially implemented)

---

## Phase 5: Policy & Governance Framework 📋

**Status:** 📋 **PLANNED**

**Objective:** Enterprise governance and compliance features

### 5.1 Policy Framework 📋

* 📋 **PolicyDefinition:** Base class for governance rules
* 📋 **Security Policies:** RBAC, data classification, access control
* 📋 **Resource Policies:** Rate limiting, quota management, SLA enforcement
* 📋 **Validation Policies:** Schema enforcement, content filtering
* 📋 **Custom Policies:** User-defined business rules
* 📋 **Policy Registry:** Discovery and management of policies
* 📋 **Policy Enforcement:** Pre/post execution validation

### 5.2 Policy Integration Points 📋

* 📋 **Pre-execution validation:** Check policies before pipeline runs
* 📋 **Node-level policies:** Per-node policy enforcement
* 📋 **Post-execution auditing:** Log and validate results
* 📋 **Cross-workflow governance:** Policies across pipelines

---

## Phase 6: Multi-Orchestrator Support 📋

**Status:** 📋 **PLANNED**

**Objective:** Specialized orchestrators for different workload types

### 6.1 Orchestrator Configurations 📋

* 📋 **Resource-optimized:** CPU, memory, throughput specialization
* 📋 **Compliance orchestrators:** Dedicated secure execution environments
* 📋 **YAML-based config:** Declarative orchestrator definitions
* 📋 **Health monitoring:** Availability and performance tracking

### 6.2 Workload Routing 📋

* 📋 **Policy-aware routing:** Route based on policy requirements
* 📋 **Dynamic scaling:** Auto-scaling orchestrator pools
* 📋 **Load balancing:** Distribute workloads efficiently
* 📋 **Workload isolation:** Separate regulated and standard workloads

### 6.3 Policy-Orchestrator Integration 📋

* 📋 **Compliance routing:** Route regulated workloads to compliant orchestrators
* 📋 **Resource governance:** Policy-based resource allocation
* 📋 **Audit integration:** Track policy enforcement and violations
* 📋 **Multi-tenant isolation:** Policy-based workspace separation

---

## Phase 7: Cloud Integrations 📋

**Status:** 📋 **PLANNED**

**Objective:** Native cloud platform integrations

### 7.1 Azure Integration (`hexdag[azure]`) 📋

* 📋 **Azure AD:** Authentication and authorization
* 📋 **Key Vault:** Secrets management
* 📋 **Service Bus:** Message queue integration
* 📋 **Monitor:** Native observability
* 📋 **Managed Identity:** Secure credential-less access

### 7.2 AWS Integration (`hexdag[aws]`) 📋

* 📋 **IAM Roles:** Authentication and authorization
* 📋 **Secrets Manager:** Secrets management
* 📋 **SQS/SNS:** Message queue integration
* 📋 **CloudWatch:** Native observability
* 📋 **Lambda:** Serverless execution

### 7.3 GCP Integration (`hexdag[gcp]`) 📋

* 📋 **Vertex AI:** Model deployment integration
* 📋 **Pub/Sub:** Message queue integration
* 📋 **Cloud Monitoring:** Native observability
* 📋 **BigQuery:** Data warehouse integration

### 7.4 Distributed Computing 📋

* 📋 **Spark Integration (`hexdag[spark]`):** Distributed DAG execution
* 📋 **Kubernetes (`hexdag[k8s]`):** CRDs, operators, autoscaling
* 📋 **Service Mesh:** Integration with Istio, Linkerd

---

## Phase 8: Enhanced Observability 📋

**Status:** 📋 **PLANNED**

**Objective:** Unified observability and event-driven extensibility

### 8.1 Observability Enhancements 📋

* 📋 **Registry metrics:** Component usage patterns
* 📋 **Custom component monitoring:** Health checks for user extensions
* 📋 **Distributed tracing:** Cross-component execution flows
* 📋 **Dashboard integration:** Grafana-compatible metrics

### 8.2 Event Sink System 📋

* 📋 **EventRouter:** Route events to external sinks
* 📋 **Pluggable Sinks:** Kafka, Webhooks, CloudWatch, Prometheus
* 📋 **Event Correlation:** Link registry, policy, and orchestration events
* 📋 **Replay & Auditing:** Store events for compliance
* 📋 **Policy Events:** Policy violations as structured events

---

## Summary

### ✅ Currently Available

- **Core Engine:** Full DAG orchestration with async execution
- **Resolver System:** Component discovery via module paths
- **YAML Builder:** K8s-style manifests with plugin architecture
- **Environment Variables:** `${VAR}` resolution with defaults
- **Jinja2 Templating:** Dynamic YAML with context
- **Macro System:** Reusable templates with expansion
- **Multi-document YAML:** Environment-specific configurations

### 📋 Future Roadmap

- **Component Versioning:** Version management for components
- **Macro Library:** Rich collection of coordination patterns
- **Policy Framework:** Governance and compliance enforcement
- **Multi-Orchestrator:** Specialized execution environments
- **Cloud Integrations:** Azure, AWS, GCP native support
- **Enhanced Observability:** Distributed tracing and event sinks

For current implementation details, see [YAML_PIPELINES_ARCHITECTURE.md](YAML_PIPELINES_ARCHITECTURE.md).
