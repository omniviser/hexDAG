"""Port interfaces for the application."""

from hexdag.core.ports.api_call import APICall
from hexdag.core.ports.database import (
    ColumnSchema,
    ColumnType,
    DatabasePort,
    SupportsRawSQL,
    SupportsReadOnly,
    SupportsStreamingQuery,
    TableSchema,
)
from hexdag.core.ports.executor import (
    ExecutionResult,
    ExecutionTask,
    ExecutorPort,
)
from hexdag.core.ports.file_storage import FileStoragePort
from hexdag.core.ports.healthcheck import HealthStatus
from hexdag.core.ports.llm import (
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
from hexdag.core.ports.memory import Memory
from hexdag.core.ports.observer_manager import ObserverManagerPort
from hexdag.core.ports.secret import SecretPort
from hexdag.core.ports.tool_router import ToolRouter
from hexdag.core.ports.vector_search import SupportsVectorSearch

__all__ = [
    "APICall",
    "ColumnSchema",
    "ColumnType",
    "DatabasePort",
    "ExecutionResult",
    "ExecutionTask",
    "ExecutorPort",
    "FileStoragePort",
    "HealthStatus",
    "ImageContent",
    "ImageInput",
    "LLM",
    "Memory",
    "ObserverManagerPort",
    "SecretPort",
    "SupportsEmbedding",
    "SupportsFunctionCalling",
    "SupportsGeneration",
    "SupportsRawSQL",
    "SupportsReadOnly",
    "SupportsStreamingQuery",
    "SupportsUsageTracking",
    "SupportsVectorSearch",
    "SupportsVision",
    "TableSchema",
    "ToolRouter",
    "VisionMessage",
]
