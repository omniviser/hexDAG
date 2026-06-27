"""Port interfaces for the application."""

from typing import Any

from hexdag.kernel.ports.api_call import APICall
from hexdag.kernel.ports.data_store import (
    DataStore,
    SupportsCollectionStorage,
    SupportsKeyValue,
    SupportsQuery,
    SupportsSchema,
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
from hexdag.kernel.ports.dict_bridge import dict_bridge_call
from hexdag.kernel.ports.entity_state import EntityState
from hexdag.kernel.ports.executor import (
    ExecutionResult,
    ExecutionTask,
    Executor,
)
from hexdag.kernel.ports.healthcheck import HealthStatus
from hexdag.kernel.ports.llm import (
    LLM,
    BatchItemResult,
    BatchItemStatus,
    BatchResult,
    ImageContent,
    ImageInput,
    SupportsBatchGeneration,
    SupportsEmbedding,
    SupportsFunctionCalling,
    SupportsGeneration,
    SupportsStructuredOutput,
    SupportsUsageTracking,
    SupportsVision,
    VisionMessage,
)
from hexdag.kernel.ports.memory import Memory
from hexdag.kernel.ports.notification import Notification
from hexdag.kernel.ports.observer_manager import ObserverManager
from hexdag.kernel.ports.pipeline_spawner import PipelineSpawner
from hexdag.kernel.ports.secret import SecretStore
from hexdag.kernel.ports.vector_search import SupportsVectorSearch
from hexdag.kernel.ports.vfs import VFS, VFSProvider
from hexdag.kernel.tool_router import ToolRouter

# Backward-compat aliases (deprecated: use new names)
DatabasePort = Database
ExecutorPort = Executor
ObserverManagerPort = ObserverManager
SecretPort = SecretStore

# Plugin-contributed port protocols resolved lazily (kept in __all__ for
# back-compat and tooling).  Loaded on first access only, so importing this
# package never forces the plugin to import.
_PLUGIN_PORTS = ("SupportsTransactions", "SupportsSessionFactory")


def __getattr__(name: str) -> Any:  # PEP 562
    """Lazily resolve plugin-contributed port protocols (e.g. database tx)."""
    if name in _PLUGIN_PORTS:
        from hexdag.kernel.ports.registry import resolve_plugin_port

        proto = resolve_plugin_port(name)
        if proto is not None:
            return proto
        raise AttributeError(
            f"{name!r} is provided by the database plugin; "
            f"install it with `pip install hexdag-plugins[database]`"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals().keys(), *_PLUGIN_PORTS])


__all__ = [
    "APICall",
    "BatchItemResult",
    "BatchItemStatus",
    "BatchResult",
    "detect_port_type",
    "dict_bridge_call",
    "ColumnSchema",
    "ColumnType",
    "DataStore",
    "Database",
    "EntityState",
    "SupportsCollectionStorage",
    "DatabasePort",
    "Executor",
    "ExecutionResult",
    "ExecutionTask",
    "ExecutorPort",
    "HealthStatus",
    "ImageContent",
    "ImageInput",
    "LLM",
    "SupportsBatchGeneration",
    "Memory",
    "Notification",
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
    "SupportsStructuredOutput",
    "SupportsTTL",
    "SupportsTransactions",  # noqa: F822 # pyright: ignore[reportUnsupportedDunderAll]
    "SupportsUsageTracking",
    "SupportsVectorSearch",
    "SupportsVision",
    "TableSchema",
    "ToolRouter",
    "VFS",
    "VFSProvider",
    "VisionMessage",
]
