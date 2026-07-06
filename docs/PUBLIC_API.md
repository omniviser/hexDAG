# hexDAG Public API Reference

> **API Stability:** Starting with v1.0, all symbols listed in this document
> follow [Semantic Versioning](https://semver.org/). Breaking changes only
> happen in major releases (2.0, 3.0, ...). New additions may happen in minor
> releases (1.1, 1.2, ...).

---

## Version Policy

| Change type | Version bump | Example |
|---|---|---|
| Breaking removal / rename / signature change | **Major** (2.0) | Removing `PipelineRunner.run()` |
| New public symbol or optional parameter | **Minor** (1.1) | Adding `PipelineRunner.dry_run()` |
| Bug fix, performance, internal refactor | **Patch** (1.0.1) | Fixing retry logic |

**What counts as public:** Any symbol listed in `__all__` of the modules below.

**What is internal:** Anything with a leading underscore, anything not in `__all__`,
and anything imported only inside `TYPE_CHECKING` blocks.

---

## Top-Level: `from hexdag import ...`

The primary user-facing API. Intentionally small.

### Execution

| Symbol | Type | Purpose |
|---|---|---|
| `PipelineRunner` | Class | Run YAML pipelines. **The main entry point.** |
| `PipelineResult` | Dataclass | Result of a pipeline run (status, outputs, metadata) |
| `System` | Class | Multi-pipeline system with shared ports and state machines |

### Node Factories

| Symbol | Type | Purpose |
|---|---|---|
| `LLMNode` | Class | LLM prompt + response node |
| `FunctionNode` | Class | Python function node |
| `ReActAgentNode` | Class | Multi-step reasoning agent |

### Port Protocols

| Symbol | Type | Purpose |
|---|---|---|
| `LLM` | Protocol | Language model port |
| `Database` | Protocol | SQL database port |
| `APICall` | Protocol | HTTP API port |

### Adapters (Testing)

| Symbol | Type | Purpose |
|---|---|---|
| `MockLLM` | Class | Mock LLM for testing |
| `MockDatabaseAdapter` | Class | Mock database for testing |
| `MockHttpClient` | Class | Mock HTTP client for testing |
| `InMemoryMemory` | Class | In-memory key-value adapter |
| `HttpClientDriver` | Class | Real HTTP client driver |

### Templating

| Symbol | Type | Purpose |
|---|---|---|
| `PromptTemplate` | Class | Jinja2 prompt template |
| `FewShotPromptTemplate` | Class | Few-shot example template |

### Deprecated (will be removed in 2.0)

| Symbol | Use instead |
|---|---|
| `Orchestrator` | `PipelineRunner` or `hexdag.kernel.Orchestrator` |
| `DirectedGraph` | `hexdag.kernel.DirectedGraph` |
| `NodeSpec` | `hexdag.kernel.NodeSpec` |
| `YamlPipelineBuilder` | `PipelineRunner` or `hexdag.compiler.yaml_builder.YamlPipelineBuilder` |
| `resolve` | `hexdag.kernel.resolve` |
| `resolve_function` | `hexdag.kernel.resolve_function` |

---

## Kernel: `from hexdag.kernel import ...`

The comprehensive framework interface. Use this when building custom components.

### Execution

| Symbol | Purpose |
|---|---|
| `PipelineRunner` | Run YAML pipelines |
| `System` | Multi-pipeline system |
| `Orchestrator` | Low-level DAG walker |
| `OrchestratorConfig` | Orchestrator settings |
| `PortsBuilder` | Build port instances from config |
| `PortConfig` | Single port configuration |
| `PortsConfiguration` | Complete ports configuration |
| `ExecutionContext` | Async context manager for execution |

### Domain Models

| Symbol | Purpose |
|---|---|
| `DirectedGraph` | DAG data structure |
| `NodeSpec` | Node definition (function + metadata) |
| `PipelineRun` | Pipeline execution record |
| `RunStatus` | Execution status enum |
| `StateMachineConfig` | State machine definition |
| `StateTransition` | State change record |

### Port Protocols

| Symbol | Purpose |
|---|---|
| `LLM` | Language model |
| `Database` | SQL database |
| `DataStore` | Key-value / document store |
| `APICall` | HTTP API |
| `SecretStore` | Secret management |
| `Executor` | Task execution |
| `ObserverManager` | Event observation |
| `PipelineSpawner` | Child pipeline execution |

### Tool Router (`from hexdag.kernel.tool_router import ...`)

`ToolRouter` is a **concrete class**, not a port protocol. It dispatches
agent tool calls to Python functions. The framework creates it automatically
from `spec.tools` in YAML — users rarely instantiate it directly.

| Symbol | Purpose |
|---|---|
| `ToolRouter` | Dispatch agent tool calls to Python functions |
| `ToolRouterCall` | Event emitted on each tool call |
| `tool_schema_from_callable` | Generate tool schema from a function signature |

### Port Capabilities (`Supports*`)

| Symbol | Port | Purpose |
|---|---|---|
| `SupportsGeneration` | LLM | Text generation |
| `SupportsFunctionCalling` | LLM | Tool/function calling |
| `SupportsStructuredOutput` | LLM | JSON schema output |
| `SupportsEmbedding` | LLM | Text embeddings |
| `SupportsVision` | LLM | Image understanding |
| `SupportsUsageTracking` | LLM | Token usage tracking |
| `SupportsKeyValue` | DataStore | Get/set operations |
| `SupportsQuery` | DataStore/DB | Query operations |
| `SupportsTTL` | DataStore | Time-to-live |
| `SupportsSchema` | DataStore/DB | Schema introspection |
| `SupportsTransactions` | DataStore/DB | Transaction support |
| `SupportsRawSQL` | Database | Raw SQL execution |
| `SupportsVectorSearch` | DataStore | Vector similarity search |

### Services & Extension Points

| Symbol | Purpose |
|---|---|
| `Service` | Base class for business logic services |
| `tool` | Decorator: mark method as agent-callable |
| `step` | Decorator: mark method as DAG step |
| `get_service_tool_schemas` | Generate tool schemas from `@tool` methods |
| `SchemaGenerator` | Generate JSON Schema from Python signatures |
| `SecretField` | Mark adapter parameters as secrets |

### Events

| Symbol | Purpose |
|---|---|
| `Event` | Base event class |
| `NodeStarted` | Node began execution |
| `NodeCompleted` | Node finished successfully |
| `NodeFailed` | Node failed |
| `NodeCancelled` | Node cancelled (timeout) |
| `NodeSkipped` | Node skipped (when clause) |
| `WaveCompleted` | Execution wave completed |
| `PipelineStarted` | Pipeline began |
| `PipelineCompleted` | Pipeline finished |

### Context (runtime access)

| Symbol | Purpose |
|---|---|
| `get_port` | Get a port instance by name |
| `get_ports` | Get all ports |
| `get_run_id` | Get current run ID |
| `get_observer_manager` | Get the observer manager |

### Component Resolution

| Symbol | Purpose |
|---|---|
| `resolve` | Resolve a component by alias or module path |
| `resolve_function` | Resolve a Python function by module path |
| `get_builtin_aliases` | Get all built-in component aliases |
| `register_alias` | Register a custom alias |

### Exceptions

All exceptions inherit from `HexDAGError`.

| Symbol | When |
|---|---|
| `HexDAGError` | Base for all framework errors |
| `ConfigurationError` | Invalid configuration |
| `ValidationError` | Validation failure |
| `ParseError` | YAML/expression parse error |
| `NodeExecutionError` | Node failed during execution |
| `NodeTimeoutError` | Node exceeded timeout |
| `OrchestratorError` | Orchestrator-level failure |
| `PipelineRunnerError` | Pipeline runner failure |
| `ExpressionError` | Invalid expression |
| `ResolveError` | Component not found |
| `InvalidTransitionError` | Invalid state transition |

---

## Standard Library

### Nodes: `from hexdag.stdlib.nodes import ...`

| Symbol | YAML `kind` | Purpose |
|---|---|---|
| `LLMNode` | `llm_node` | LLM prompt + response |
| `FunctionNode` | `function_node` | Python function |
| `ReActAgentNode` | `re_act_agent_node` | Multi-step agent |
| `ExpressionNode` | `expression_node` | Safe expression evaluation |
| `CompositeNode` | `composite_node` | Control flow (if/while/for-each) |
| `ServiceCallNode` | `service_call_node` | Service `@step` invocation |
| `TransitionNode` | `transition_node` | Entity state transition |
| `WaitNode` | `wait_node` | Suspend until external event |
| `ApiCallNode` | `api_call_node` | HTTP API call |
| `DataNode` | `data_node` | **Deprecated.** Use `ExpressionNode`. |

### Middleware: `from hexdag.stdlib.middleware import ...`

| Symbol | Type | Purpose |
|---|---|---|
| `RetryWithBackoff` | Wrapper | Retry with exponential backoff |
| `RateLimiter` | Wrapper | Token-bucket rate limiting |
| `CircuitBreaker` | Wrapper | Failure threshold -> open/half-open/closed |
| `ResponseCache` | Wrapper | LRU response cache |
| `Timeout` | Wrapper | Wall-clock timeout |
| `RoundRobin` | Wrapper | Load balance across adapters; declarative via port `adapters:` list + `strategy: round_robin`/`failover` (see [GUIDE.md > Multi-Adapter Pools](GUIDE.md#multi-adapter-pools)) |
| `BatchGeneration` | Wrapper | Batch generation control |
| `DistributedCache` | Wrapper | External store cache |
| `ResourceAccountingObserver` | Observer | Track tokens/calls, enforce limits |
| `compose` | Function | Stack middleware layers |

---

## API Layer: `from hexdag.api import ...`

### Stable (core, build-time)

| Module | Purpose |
|---|---|
| `components` | Component discovery (list_nodes, list_adapters, etc.) |
| `validation` | Pipeline validation |
| `pipeline` | YAML manipulation (init, add/remove/update nodes) |
| `documentation` | Syntax reference, guides |
| `export` | Project export |

### Moving to hexdag-brain (runtime)

These modules will move to `hexdag-brain` in a future release.
They remain importable from `hexdag.api` until then.

| Module | Purpose | Why brain |
|---|---|---|
| `execution` | Run pipelines | Runtime, not authoring |
| `systems` | Run system, transition entity | Runtime operations |
| `processes` | Spawn, schedule, list runs | Runtime management |
| `logs` | Query execution logs | Runtime introspection |

### Parked

| Module | Status |
|---|---|
| `vfs` | Parked. Not recommended for new code. |

---

## CLI Commands

| Command | Purpose |
|---|---|
| `hexdag init` | Initialize a new project |
| `hexdag validate` | Validate YAML pipelines |
| `hexdag lint` | Lint for best practices |
| `hexdag explain` | Explain nodes, adapters, syntax |
| `hexdag build` | Build pipelines |
| `hexdag create` | Create from templates |
| `hexdag generate-types` | Generate TypeScript/JSON Schema |
| `hexdag docs` | Generate and serve docs |
| `hexdag pipeline` | Pipeline subcommands |
| `hexdag studio` | Launch visual editor |
| `hexdag plugins` | Manage plugins |

---

## MCP Server Tools

Build-time tools exposed via Model Context Protocol:

| Tool | Purpose |
|---|---|
| `list_nodes` | List all node types with schemas |
| `list_adapters` | List adapters, filter by port type |
| `list_tools` | List agent tools |
| `list_macros` | List reusable macros |
| `list_tags` | List YAML custom tags |
| `get_component_schema` | Detailed schema for a component |
| `get_syntax_reference` | YAML syntax docs |
| `explain_yaml_structure` | YAML structure docs |
| `get_type_reference` | Type system docs |
| `get_custom_adapter_guide` | Adapter authoring guide |
| `get_custom_node_guide` | Node authoring guide |
| `get_custom_tool_guide` | Tool authoring guide |
| `get_extension_guide` | Extension overview |
| `validate_yaml_pipeline` | Validate YAML (strict) |
| `validate_yaml_pipeline_lenient` | Validate YAML (lenient) |
| `init_pipeline` | Create empty pipeline |
| `add_node_to_pipeline` | Add a node |
| `remove_node_from_pipeline` | Remove a node |
| `update_node_config` | Update node config |
| `list_pipeline_nodes` | List nodes in pipeline |
| `generate_pipeline_template` | Generate from node types |
| `build_yaml_pipeline_interactive` | Build from structured input |
