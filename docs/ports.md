# Ports Reference

This page documents all registered ports in HexDAG.

## Overview

Total ports: **10**

## Namespace: `core`

### `api_call`

Base protocol for making external API calls.

    This is a fundamental protocol that provides a standard interface for
    making HTTP/REST API calls. Other protocols (like LLM) can inherit from
    this to indicate they also support API call functionality.

    Implementations can use requests, httpx, aiohttp, or any HTTP client.

**Metadata:**

- **Type:** `port`
- **Namespace:** `core`
- **Name:** `api_call`

---

### `database`

Port interface for accessing database schema and metadata.

    This port abstracts access to database systems, allowing the analytics engine to work with
    different database backends. Implementations may use direct connections (psycopg2, SQLAlchemy)
    or REST APIs for cloud databases (Snowflake, BigQuery, etc.).

    Optional Methods
    ----------------
    Adapters may optionally implement:
    - ahealth_check(): Verify database connectivity and query execution

**Metadata:**

- **Type:** `port`
- **Namespace:** `core`
- **Name:** `database`

---

### `executor`

Port interface for DAG node execution strategies.

    This port abstracts the execution backend, allowing the orchestrator
    to delegate node execution to different implementations:

    - **LocalExecutor**: In-process async execution (default)
    - **CeleryExecutor**: Distributed execution via Celery task queue
    - **AzureFunctionsExecutor**: Serverless execution via Azure Functions

    The port provides a consistent interface regardless of where/how
    nodes are actually executed.

    Lifecycle
    ---------
    Executors may implement optional setup/cleanup methods:
    - asetup(): Initialize resources (connections, workers, etc.)
    - aclose(): Cleanup resources (called automatically by orchestrator)

    Optional Methods
    ----------------
    Adapters may optionally implement:
    - asetup(): Initialize executor resources before first use
    - aclose(): Cleanup executor resources after execution completes

**Metadata:**

- **Type:** `port`
- **Namespace:** `core`
- **Name:** `executor`

---

### `file_storage`

Port for file storage operations.

    Provides a unified interface for local and cloud file storage.

**Metadata:**

- **Type:** `port`
- **Namespace:** `core`
- **Name:** `file_storage`

---

### `llm`

Port interface for Large Language Models (LLMs).


    LLMs provide natural language generation capabilities. Implementations
    may use various backends (OpenAI, Anthropic, local models, etc.) but
    must provide the aresponse method for generating text from messages.

    Optional Protocols
    ------------------
    Adapters may optionally implement:
    - **SupportsGeneration**: Text generation via `aresponse()`
    - **SupportsFunctionCalling**: Native tool calling via `aresponse_with_tools()`
    - **SupportsVision**: Multimodal vision via `aresponse_with_vision()`
    - **SupportsEmbedding**: Embedding generation via `aembed()`
    - **SupportsUsageTracking**: Token usage tracking via `get_last_usage()`
    - ahealth_check(): Verify LLM API connectivity and availability

    SupportsUsageTracking
    ---------------------
    Adapters that track token usage implement `get_last_usage() -> TokenUsage | None`.
    This enables the `CostProfilerObserver` to compute per-node costs without
    changing the `SupportsGeneration.aresponse() -> str | None` return type.

    ```python
    from hexdag.core.ports.llm import SupportsUsageTracking, TokenUsage

    class MyAdapter(LLM, SupportsUsageTracking):
        def get_last_usage(self) -> TokenUsage | None:
            return self._last_usage  # Set after each API call
    ```

**Metadata:**

- **Type:** `port`
- **Namespace:** `core`
- **Name:** `llm`

---

### `memory`

Protocol for long-term memory storage and retrieval.

    Optional Methods
    ----------------
    Adapters may optionally implement:
    - ahealth_check(): Verify storage backend connectivity and availability

**Metadata:**

- **Type:** `port`
- **Namespace:** `core`
- **Name:** `memory`

---

### `observer_manager`

Port interface for event observation systems.

    Key Safety Guarantees:
    - Observers must be READ-ONLY and cannot affect execution
    - Observer failures must not crash the pipeline (fault isolation)
    - Fire-and-forget pattern with async, non-blocking execution
    - Event type filtering for performance optimization
    - Configurable concurrency control and timeouts
    - Optional weak reference support for memory management

**Metadata:**

- **Type:** `port`
- **Namespace:** `core`
- **Name:** `observer_manager`

---

### `policy_manager`

Port for managing execution control policies.

    Provides subscription-based policy management with automatic cleanup
    using weak references and categorization by subscriber type.

**Metadata:**

- **Type:** `port`
- **Namespace:** `core`
- **Name:** `policy_manager`

---

### `secret`

Port interface for secret/credential management systems.

    This port abstracts access to secret management services like:
    - Azure KeyVault
    - AWS Secrets Manager
    - HashiCorp Vault
    - Google Secret Manager
    - Environment variables

    Secrets are returned as Secret[str] objects to prevent accidental logging.

    Optional Methods
    ----------------
    Adapters may optionally implement:
    - ahealth_check(): Verify secret service connectivity and authentication

**Metadata:**

- **Type:** `port`
- **Namespace:** `core`
- **Name:** `secret`

---

### `tool_router`

Protocol for routing tool calls.

**Metadata:**

- **Type:** `port`
- **Namespace:** `core`
- **Name:** `tool_router`

---
