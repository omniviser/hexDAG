"""Port interfaces for the application."""

from hexdag.core.ports.api_call import APICall
from hexdag.core.ports.database import (
    ColumnSchema,
    ColumnType,
    DatabasePort,
    SupportsRawSQL,
    SupportsReadOnly,
    SupportsStreamingQuery,
    SupportsVectorSearch,
    TableSchema,
)
from hexdag.core.ports.embedding import Embedding
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
    SupportsVision,
    VisionMessage,
)
from hexdag.core.ports.memory import Memory
from hexdag.core.ports.observer_manager import ObserverManagerPort
from hexdag.core.ports.policy_manager import PolicyManagerPort
from hexdag.core.ports.secret import SecretPort
from hexdag.core.ports.tool_router import ToolRouter

__all__ = [
    "APICall",
    "ColumnSchema",
    "ColumnType",
    "DatabasePort",
    "Embedding",  # Deprecated - kept for backward compatibility
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
    "PolicyManagerPort",
    "SecretPort",
    "SupportsEmbedding",  # New - unified embedding support in LLM port
    "SupportsFunctionCalling",
    "SupportsGeneration",  # New - text generation support in LLM port
    "SupportsRawSQL",
    "SupportsReadOnly",
    "SupportsStreamingQuery",
    "SupportsVectorSearch",
    "SupportsVision",
    "TableSchema",
    "ToolRouter",
    "VisionMessage",
]
