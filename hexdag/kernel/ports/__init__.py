"""Port interfaces for the application."""

from hexdag.kernel.ports.api_call import APICall
from hexdag.kernel.ports.data_store import (
    DataStore,
    SupportsCollectionStorage,
    SupportsKeyValue,
    SupportsQuery,
    SupportsSchema,
    SupportsTransactions,
    SupportsTTL,
)
from hexdag.kernel.ports.database import (
    ColumnSchema,
    ColumnType,
    Database,
    SupportsRawSQL,
    SupportsReadOnly,
    SupportsStreamingQuery,
    TableSchema,
)
from hexdag.kernel.ports.detection import detect_port_type
from hexdag.kernel.ports.executor import (
    ExecutionResult,
    ExecutionTask,
    Executor,
)
from hexdag.kernel.ports.file_storage import FileStorage
from hexdag.kernel.ports.healthcheck import HealthStatus
from hexdag.kernel.ports.llm import (
    LLM,
    ImageContent,
    ImageInput,
    SupportsEmbedding,
    SupportsFunctionCalling,
    SupportsGeneration,
    SupportsUsageTracking,
    SupportsVision,
    VisionMessage,
)
from hexdag.kernel.ports.memory import Memory
from hexdag.kernel.ports.observer_manager import ObserverManager
from hexdag.kernel.ports.pipeline_spawner import PipelineSpawner
from hexdag.kernel.ports.secret import SecretStore
from hexdag.kernel.ports.tool_router import ToolRouter
from hexdag.kernel.ports.vector_search import SupportsVectorSearch
from hexdag.kernel.ports.vfs import VFS, VFSProvider

# Backward-compat aliases (deprecated: use new names)
DatabasePort = Database
ExecutorPort = Executor
FileStoragePort = FileStorage
ObserverManagerPort = ObserverManager
SecretPort = SecretStore

__all__ = [
    "APICall",
    "detect_port_type",
    "ColumnSchema",
    "ColumnType",
    "DataStore",
    "Database",
    "SupportsCollectionStorage",
    "DatabasePort",
    "Executor",
    "ExecutionResult",
    "ExecutionTask",
    "ExecutorPort",
    "FileStorage",
    "FileStoragePort",
    "HealthStatus",
    "ImageContent",
    "ImageInput",
    "LLM",
    "Memory",
    "ObserverManager",
    "ObserverManagerPort",
    "PipelineSpawner",
    "SecretStore",
    "SecretPort",
    "SupportsEmbedding",
    "SupportsFunctionCalling",
    "SupportsGeneration",
    "SupportsKeyValue",
    "SupportsQuery",
    "SupportsRawSQL",
    "SupportsReadOnly",
    "SupportsSchema",
    "SupportsStreamingQuery",
    "SupportsTTL",
    "SupportsTransactions",
    "SupportsUsageTracking",
    "SupportsVectorSearch",
    "SupportsVision",
    "TableSchema",
    "ToolRouter",
    "VFS",
    "VFSProvider",
    "VisionMessage",
]
