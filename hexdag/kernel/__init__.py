"""hexDAG Kernel — the syscall table.

This module defines the **public API** of the kernel. User-space code
(``hexdag.api``, ``hexdag.cli``, ``hexdag.mcp_server``, and end-user
applications) should import exclusively from ``hexdag.kernel``, never
from kernel submodules.

Kernel-space code (``hexdag.kernel.*``, ``hexdag.stdlib.*``,
``hexdag.compiler.*``, ``hexdag.drivers.*``) may freely import
from kernel submodules.

The exports are grouped by category, mirroring OS concepts:
- Pipeline execution (primary syscalls)
- Domain types (kernel structs)
- Port protocols (device interfaces)
- Context (syscall interface for node code)
- Component resolution
- Exceptions (error codes)
- Protocols (structural typing)
- Extension points
- Events (kernel notifications)
- Schema introspection
- Linting
- Discovery
- Logging
- Types
- Prompt templating
- Expression evaluation
- Data models
- VFS domain
"""

# ============================================================================
# 1. Pipeline Execution (primary syscalls)
# ============================================================================

# ============================================================================
# 8. Extension Points
# ============================================================================
from hexdag.kernel.configurable import (
    ConfigurableMacro,
    MacroConfig,
    SecretField,
)

# ============================================================================
# 4. Context (syscall interface for node code — getters only)
# ============================================================================
# Keep setters importable for backward compat but NOT in __all__
from hexdag.kernel.context import (  # noqa: F401
    ExecutionContext,
    clear_execution_context,
    get_observer_manager,
    get_port,
    get_ports,
    get_run_id,
    set_observer_manager,
    set_ports,
    set_run_id,
)

# ============================================================================
# 12. Discovery
# ============================================================================
from hexdag.kernel.discovery import (
    discover_adapters_in_package,
    discover_classes_in_module,
    discover_macros_in_module,
    discover_plugins,
    discover_tools_in_module,
    discover_user_modules,
    discover_user_plugins,
    set_user_plugin_paths,
)

# ============================================================================
# 2. Domain Types (kernel structs)
# ============================================================================
from hexdag.kernel.domain import DirectedGraph, NodeSpec
from hexdag.kernel.domain.entity_state import StateMachineConfig, StateTransition
from hexdag.kernel.domain.pipeline_run import PipelineRun, RunStatus
from hexdag.kernel.domain.scheduled_task import ScheduledTask, ScheduleType, TaskStatus
from hexdag.kernel.domain.vfs import DirEntry, EntryType, StatResult

# ============================================================================
# 6. Exceptions (error codes)
# ============================================================================
from hexdag.kernel.exceptions import (
    BodyExecutorError,
    ComponentInstantiationError,
    ConfigurationError,
    CycleDetectedError,
    DependencyError,
    DirectedGraphError,
    DuplicateNodeError,
    ExpressionError,
    HexDAGError,
    HttpClientError,
    IncludeTagError,
    InvalidTransitionError,
    MissingDependencyError,
    MissingVariableError,
    NodeExecutionError,
    NodeTimeoutError,
    NodeValidationError,
    OrchestratorError,
    ParseError,
    PipelineRunnerError,
    PromptTemplateError,
    ResolveError,
    ResourceNotFoundError,
    SchemaCompatibilityError,
    TypeMismatchError,
    ValidationError,
    VFSError,
    YamlPipelineBuilderError,
)

# ============================================================================
# 16. Expression Evaluation
# ============================================================================
from hexdag.kernel.expression_parser import (
    ALLOWED_FUNCTIONS,
    compile_expression,
    evaluate_expression,
)
from hexdag.kernel.lib_base import HexDAGLib

# ============================================================================
# 11. Linting
# ============================================================================
from hexdag.kernel.linting.models import LintReport, LintViolation
from hexdag.kernel.linting.pipeline_rules import ALL_PIPELINE_RULES, run_pipeline_rules
from hexdag.kernel.linting.rules import LintRule, run_rules

# ============================================================================
# 13. Logging
# ============================================================================
from hexdag.kernel.logging import configure_logging, get_logger

# ============================================================================
# 17. Data Models
# ============================================================================
from hexdag.kernel.models import (
    DataReference,
    FileData,
    InMemoryData,
    JsonData,
    ListData,
    StringData,
)

# ============================================================================
# 9. Events (kernel notifications)
# ============================================================================
from hexdag.kernel.orchestration.events import (
    CheckpointRestored,
    CheckpointSaved,
    Event,
    HealthCheckCompleted,
    LLMPromptSent,
    LLMResponseReceived,
    NodeCancelled,
    NodeCompleted,
    NodeFailed,
    NodeSkipped,
    NodeStarted,
    PipelineCancelled,
    PipelineCompleted,
    PipelineStarted,
    PolicyEvaluated,
    PolicyFallback,
    PolicyRetry,
    PolicySkipped,
    PolicyTriggered,
    ToolCalled,
    ToolCompleted,
    WaveCompleted,
    WaveStarted,
)
from hexdag.kernel.orchestration.models import (
    OrchestratorConfig,
    PortConfig,
    PortsConfiguration,
)
from hexdag.kernel.orchestration.orchestrator import Orchestrator

# ============================================================================
# 15. Prompt Templating
# ============================================================================
from hexdag.kernel.orchestration.prompt import FewShotPromptTemplate, PromptTemplate
from hexdag.kernel.pipeline_runner import PipelineRunner

# ============================================================================
# 3. Port Protocols (device interfaces)
# ============================================================================
from hexdag.kernel.ports import (
    LLM,
    VFS,
    APICall,
    ColumnSchema,
    ColumnType,
    Database,
    DataStore,
    ExecutionResult,
    ExecutionTask,
    Executor,
    FileStorage,
    HealthStatus,
    ImageContent,
    ImageInput,
    Memory,
    ObserverManager,
    PipelineSpawner,
    SecretStore,
    SupportsCollectionStorage,
    SupportsEmbedding,
    SupportsFunctionCalling,
    SupportsGeneration,
    SupportsKeyValue,
    SupportsQuery,
    SupportsRawSQL,
    SupportsReadOnly,
    SupportsSchema,
    SupportsStreamingQuery,
    SupportsTransactions,
    SupportsTTL,
    SupportsUsageTracking,
    SupportsVectorSearch,
    SupportsVision,
    TableSchema,
    ToolRouter,
    VFSProvider,
    VisionMessage,
)
from hexdag.kernel.ports.detection import detect_port_type
from hexdag.kernel.ports.llm import Message, MessageList
from hexdag.kernel.ports_builder import PortsBuilder

# ============================================================================
# 7. Protocols (structural typing)
# ============================================================================
from hexdag.kernel.protocols import (
    ComponentWithExecute,
    ConfigurablePort,
    DictConvertible,
    HealthCheckable,
    SchemaProvider,
    has_execute_method,
    is_dict_convertible,
    is_schema_type,
    to_dict,
)

# ============================================================================
# 5. Component Resolution
# ============================================================================
from hexdag.kernel.resolver import (
    get_builtin_aliases,
    get_registered_aliases,
    register_alias,
    resolve,
    resolve_function,
    unregister_alias,
)

# ============================================================================
# 10. Schema Introspection
# ============================================================================
from hexdag.kernel.schema import SchemaGenerator
from hexdag.kernel.service import Service, get_service_tool_schemas, step, tool

# ============================================================================
# 14. Types
# ============================================================================
from hexdag.kernel.types import Secret

# ============================================================================
# __all__ — the complete syscall table
# ============================================================================

__all__ = [
    # -- 1. Pipeline Execution --
    "PipelineRunner",
    "Orchestrator",
    "OrchestratorConfig",
    "PortsBuilder",
    "PortConfig",
    "PortsConfiguration",
    # -- 2. Domain Types --
    "DirectedGraph",
    "NodeSpec",
    "PipelineRun",
    "RunStatus",
    "ScheduledTask",
    "ScheduleType",
    "TaskStatus",
    "StateMachineConfig",
    "StateTransition",
    "DirEntry",
    "EntryType",
    "StatResult",
    # -- 3. Port Protocols --
    "LLM",
    "Database",
    "DataStore",
    "APICall",
    "ToolRouter",
    "Memory",
    "FileStorage",
    "SecretStore",
    "Executor",
    "ObserverManager",
    "PipelineSpawner",
    "VFS",
    "VFSProvider",
    "HealthStatus",
    "detect_port_type",
    # Port capability sub-protocols
    "SupportsGeneration",
    "SupportsFunctionCalling",
    "SupportsEmbedding",
    "SupportsVision",
    "SupportsUsageTracking",
    "SupportsKeyValue",
    "SupportsQuery",
    "SupportsTTL",
    "SupportsSchema",
    "SupportsTransactions",
    "SupportsCollectionStorage",
    "SupportsRawSQL",
    "SupportsReadOnly",
    "SupportsStreamingQuery",
    "SupportsVectorSearch",
    # Port supporting models
    "ExecutionResult",
    "ExecutionTask",
    "ImageContent",
    "ImageInput",
    "VisionMessage",
    "ColumnSchema",
    "ColumnType",
    "TableSchema",
    "Message",
    "MessageList",
    # -- 4. Context --
    "ExecutionContext",
    "get_port",
    "get_ports",
    "get_run_id",
    "get_observer_manager",
    # -- 5. Component Resolution --
    "resolve",
    "resolve_function",
    "register_alias",
    "unregister_alias",
    "get_registered_aliases",
    "get_builtin_aliases",
    # -- 6. Exceptions --
    "HexDAGError",
    "ConfigurationError",
    "ValidationError",
    "ParseError",
    "ResourceNotFoundError",
    "DependencyError",
    "TypeMismatchError",
    "OrchestratorError",
    "NodeExecutionError",
    "NodeTimeoutError",
    "BodyExecutorError",
    "PromptTemplateError",
    "MissingVariableError",
    "PipelineRunnerError",
    "ExpressionError",
    "ResolveError",
    "NodeValidationError",
    "DirectedGraphError",
    "CycleDetectedError",
    "MissingDependencyError",
    "DuplicateNodeError",
    "SchemaCompatibilityError",
    "YamlPipelineBuilderError",
    "ComponentInstantiationError",
    "IncludeTagError",
    "HttpClientError",
    "InvalidTransitionError",
    "VFSError",
    # -- 7. Protocols --
    "ComponentWithExecute",
    "ConfigurablePort",
    "DictConvertible",
    "HealthCheckable",
    "SchemaProvider",
    "has_execute_method",
    "is_dict_convertible",
    "is_schema_type",
    "to_dict",
    # -- 8. Extension Points --
    "SecretField",
    "ConfigurableMacro",
    "MacroConfig",
    "HexDAGLib",
    "Service",
    "tool",
    "step",
    "get_service_tool_schemas",
    # -- 9. Events --
    "Event",
    "NodeStarted",
    "NodeCompleted",
    "NodeFailed",
    "NodeCancelled",
    "NodeSkipped",
    "WaveStarted",
    "WaveCompleted",
    "PipelineStarted",
    "PipelineCompleted",
    "PipelineCancelled",
    "LLMPromptSent",
    "LLMResponseReceived",
    "ToolCalled",
    "ToolCompleted",
    "PolicyEvaluated",
    "PolicyTriggered",
    "PolicySkipped",
    "PolicyFallback",
    "PolicyRetry",
    "CheckpointSaved",
    "CheckpointRestored",
    "HealthCheckCompleted",
    # -- 10. Schema Introspection --
    "SchemaGenerator",
    # -- 11. Linting --
    "LintRule",
    "LintReport",
    "LintViolation",
    "run_rules",
    "run_pipeline_rules",
    "ALL_PIPELINE_RULES",
    # -- 12. Discovery --
    "discover_adapters_in_package",
    "discover_tools_in_module",
    "discover_macros_in_module",
    "discover_classes_in_module",
    "discover_plugins",
    "discover_user_modules",
    "discover_user_plugins",
    "set_user_plugin_paths",
    # -- 13. Logging --
    "get_logger",
    "configure_logging",
    # -- 14. Types --
    "Secret",
    # -- 15. Prompt Templating --
    "PromptTemplate",
    "FewShotPromptTemplate",
    # -- 16. Expression Evaluation --
    "ALLOWED_FUNCTIONS",
    "compile_expression",
    "evaluate_expression",
    # -- 17. Data Models --
    "DataReference",
    "FileData",
    "InMemoryData",
    "StringData",
    "JsonData",
    "ListData",
]
