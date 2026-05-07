# Ports Reference

Port protocols define the interfaces that adapters must implement.
hexDAG follows hexagonal architecture — ports live in the kernel,
adapters in stdlib or plugins.

## Overview

| Port | Module |
|------|--------|
| `LLM` | `hexdag.kernel.ports.llm` |
| `Database` | `hexdag.kernel.ports.database` |
| `Memory` | `hexdag.kernel.ports.memory` |
| `ToolRouter` | `hexdag.kernel.ports.tool_router` |
| `APICall` | `hexdag.kernel.ports.api_call` |
| `FileStorage` | `hexdag.kernel.ports.file_storage` |
| `SecretStore` | `hexdag.kernel.ports.secret` |
| `Executor` | `hexdag.kernel.ports.executor` |

---

### `LLM`

**Module:** `hexdag.kernel.ports.llm`

Port interface for Large Language Models (LLMs).

---

### `Database`

**Module:** `hexdag.kernel.ports.database`

Port interface for accessing database schema and metadata.

**Methods:**

- `aexecute_query(self, query: 'str', params: 'dict[str, Any] | None' = None) -> 'list[dict[str, Any]]'`
- `aget_table_schemas(self) -> 'dict[str, dict[str, Any]]'`

---

### `Memory`

**Module:** `hexdag.kernel.ports.memory`

Protocol for long-term memory storage and retrieval.

**Methods:**

- `aget(self, key: str) -> Any`
- `ahealth_check(self) -> 'HealthStatus'`
- `aset(self, key: str, value: Any) -> None`

---

### `ToolRouter`

**Module:** `hexdag.kernel.ports.tool_router`

Concrete tool router that wraps plain Python functions.

**Methods:**

- `acall_tool(self, tool_name: 'str', params: 'dict[str, Any]') -> 'Any'`
- `add_tool(self, name: 'str', fn: 'Callable[..., Any]') -> 'None'`
- `add_tools_from(self, other: 'ToolRouter') -> 'None'`
- `aget_all_tool_schemas(self) -> 'dict[str, dict[str, Any]]'`
- `aget_available_tools(self) -> 'list[str]'`
- `aget_tool_schema(self, tool_name: 'str') -> 'dict[str, Any]'`
- `ahealth_check(self) -> 'dict[str, Any]'`
- `get_all_tool_schemas(self) -> 'dict[str, dict[str, Any]]'`
- `get_available_tools(self) -> 'list[str]'`
- `get_tool_schema(self, tool_name: 'str') -> 'dict[str, Any]'`

---

### `APICall`

**Module:** `hexdag.kernel.ports.api_call`

Base protocol for making external API calls.

**Methods:**

- `adelete(self, url: str, headers: dict[str, str] | None = None, **kwargs: Any) -> dict[str, typing.Any]`
- `aget(self, url: str, headers: dict[str, str] | None = None, params: dict[str, typing.Any] | None = None, **kwargs: Any) -> dict[str, typing.Any]`
- `apost(self, url: str, json: dict[str, typing.Any] | None = None, data: typing.Any | None = None, headers: dict[str, str] | None = None, **kwargs: Any) -> dict[str, typing.Any]`
- `aput(self, url: str, json: dict[str, typing.Any] | None = None, data: typing.Any | None = None, headers: dict[str, str] | None = None, **kwargs: Any) -> dict[str, typing.Any]`
- `arequest(self, method: str, url: str, **kwargs: Any) -> dict[str, typing.Any]`

---

### `FileStorage`

**Module:** `hexdag.kernel.ports.file_storage`

Port for file storage operations.

**Methods:**

- `adelete(self, remote_path: str) -> dict`
- `adownload(self, remote_path: str, local_path: str) -> dict`
- `aexists(self, remote_path: str) -> bool`
- `aget_metadata(self, remote_path: str) -> dict`
- `ahealth_check(self) -> hexdag.kernel.ports.healthcheck.HealthStatus`
- `alist(self, prefix: str = '') -> list[str]`
- `aupload(self, local_path: str, remote_path: str) -> dict`

---

### `SecretStore`

**Module:** `hexdag.kernel.ports.secret`

Port interface for secret/credential management systems.

**Methods:**

- `aget_secret(self, key: str) -> 'Secret'`
- `ahealth_check(self) -> 'HealthStatus'`
- `alist_secret_names(self) -> list[str]`
- `aload_secrets_to_memory(self, memory: 'Memory', prefix: str = 'secret:', keys: list[str] | None = None) -> dict[str, str]`
- `load_to_environ(self, keys: list[str] | None = None, prefix: str = '', overwrite: bool = False) -> dict[str, str]`

---

### `Executor`

**Module:** `hexdag.kernel.ports.executor`

Port interface for DAG node execution strategies.

**Methods:**

- `aclose(self) -> None`
- `aexecute_node(self, task: hexdag.kernel.ports.executor.ExecutionTask) -> hexdag.kernel.ports.executor.ExecutionResult`
- `aexecute_wave(self, tasks: list[hexdag.kernel.ports.executor.ExecutionTask]) -> dict[str, hexdag.kernel.ports.executor.ExecutionResult]`
- `asetup(self) -> None`

---
