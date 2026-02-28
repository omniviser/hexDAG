"""Pipeline configuration models for YAML-based component configuration.

This module provides Pydantic models for declaring and configuring:
- Global ports (adapters)
- Global policies
- Per-type port defaults (type_ports)
- Per-node port/policy overrides
- Custom sanitized types
- Aliases and schema declarations

These models are the **single source of truth** for both schema generation
(``generate_schemas.py``) and runtime builder code (``yaml_builder.py``).

Note
----
All configuration dictionaries use native YAML dict format.
Formats:
- ports: port_name -> {namespace: str, name: str, params: dict}
- type_ports: node_type -> {port_name -> {namespace: str, name: str, params: dict}}
- policies: policy_name -> {namespace: str, name: str, params: dict}
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from hexdag.kernel.config.models import DefaultCaps, DefaultLimits
    from hexdag.kernel.orchestration.models import OrchestratorConfig


class CustomTypeConfig(BaseModel):
    """Custom sanitized type definition for output_schema validation."""

    model_config = ConfigDict(extra="forbid")

    base: Literal["str", "int", "float", "bool", "Decimal"] = Field(
        description="Base type (str, int, float, bool, Decimal)"
    )
    pattern: str | None = Field(default=None, description="Regex pattern for validation")
    nulls: list[str] | None = Field(default=None, description="Values to treat as null")
    strip: bool | None = Field(default=None, description="Strip whitespace")
    trim: bool | None = Field(default=None, description="Trim whitespace")
    upper: bool | None = Field(default=None, description="Convert to uppercase")
    lower: bool | None = Field(default=None, description="Convert to lowercase")
    max_length: int | None = Field(default=None, description="Maximum string length")
    default: Any | None = Field(default=None, description="Default value if null")
    clamp: list[float] | None = Field(
        default=None,
        description="[min, max] bounds for numeric types",
        min_length=2,
        max_length=2,
    )
    true_values: list[str] | None = Field(
        default=None,
        description="Values to treat as true (bool type)",
    )
    false_values: list[str] | None = Field(
        default=None,
        description="Values to treat as false (bool type)",
    )
    description: str | None = Field(
        default=None,
        description="Human-readable description of the type",
    )


class BaseNodeConfig(BaseModel):
    """Base node-level YAML properties available on every node type.

    This model is the **single source of truth** for the common node fields.
    It is used in three places:

    1. **Schema generation** — ``generate_schemas.py`` calls
       ``model_json_schema()`` to produce the base ``Node`` definition.
    2. **Builder** — ``yaml_builder._build_graph()`` calls
       ``BaseNodeConfig.from_node_config()`` to extract these fields
       from the raw YAML dict with type validation.
    3. **Validator** — ``yaml_validator.py`` reads ``wait_for`` from the
       parsed model rather than raw dict access.
    """

    model_config = ConfigDict(extra="ignore")

    input_mapping: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Field mapping from upstream nodes or pipeline input."
            " Keys are target field names, values use"
            " node_name.field or $input.field syntax."
            " Values can also be nested dicts whose leaf strings"
            " are resolved recursively."
        ),
    )
    dependencies: list[str] | None = Field(
        default=None,
        description=(
            "Explicit execution dependencies (deprecated — prefer"
            " input_mapping or wait_for; deps are auto-inferred"
            " from data references)"
        ),
    )
    wait_for: list[str] | None = Field(
        default=None,
        description=(
            "Ordering-only dependencies — run after these nodes"
            " complete but do not consume their data"
        ),
    )

    @classmethod
    def from_node_config(cls, node_config: dict[str, Any]) -> Self:
        """Extract base fields from a raw YAML node config dict.

        Handles the fact that ``dependencies`` and ``wait_for`` live at
        the node level, while ``input_mapping`` lives inside ``spec``.
        Also normalises scalar ``wait_for`` values to a list.
        """
        spec = node_config.get("spec", {})

        # Normalise wait_for: YAML allows a bare string
        wait_for = node_config.get("wait_for")
        if isinstance(wait_for, str):
            wait_for = [wait_for]

        return cls.model_validate({
            "dependencies": node_config.get("dependencies") or spec.get("dependencies"),
            "wait_for": wait_for,
            "input_mapping": spec.get("input_mapping"),
        })


class PipelineConfig(BaseModel):
    """Complete pipeline configuration including ports and policies.

    This model represents the full configuration extracted from YAML
    that will be used to instantiate and configure the orchestrator.

    Attributes
    ----------
    ports : dict[str, dict[str, Any]]
        Global port (adapter) configurations using native YAML dict format.
        Format: {namespace: str, name: str, params: dict}
    type_ports : dict[str, dict[str, dict[str, Any]]]
        Per-type port defaults. Maps node_type -> {port_name -> dict spec}
    policies : dict[str, dict[str, Any]]
        Global policy configurations using native YAML dict format.
        Format: {namespace: str, name: str, params: dict}
    metadata : dict[str, Any]
        Pipeline metadata (name, description, version, etc.)
    nodes : list[dict[str, Any]]
        Node specifications (handled by existing builder)

    Examples
    --------
    ```yaml
    apiVersion: hexdag.omniviser.io/v1alpha1
    kind: Pipeline
    metadata:
      name: my-pipeline
      description: Example pipeline

    spec:
      # Global ports (adapters) - native YAML dict format
      ports:
        llm:
          namespace: core
          name: openai
          params:
            model: gpt-4
            temperature: 0.7
            api_key: ${OPENAI_API_KEY}
        database:
          namespace: core
          name: postgres
          params:
            connection_string: ${DB_URL}

      # Per-type port defaults
      type_ports:
        agent:
          llm:
            namespace: core
            name: anthropic
            params:
              model: claude-3-5-sonnet

      # Global policies - native YAML dict format
      policies:
        retry:
          name: retry
          params:
            max_retries: 3
        timeout:
          name: timeout
          params:
            timeout_seconds: 300

      # Nodes
      nodes:
        - kind: core:agent_node
          metadata:
            name: researcher
          spec:
            initial_prompt_template: "Research: {{topic}}"
            max_steps: 10
            # Node-level port override
            ports:
              llm:
                namespace: core
                name: openai
                params:
                  model: gpt-4o
            # Node-level policy override
            policies:
              timeout:
                name: timeout
                params:
                  timeout_seconds: 600
    ```
    """

    model_config = ConfigDict(extra="allow")  # Allow additional fields for extensibility

    # Global configuration
    ports: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Port adapters configuration (LLM, database, memory)",
    )
    type_ports: dict[str, dict[str, dict[str, Any]]] = Field(
        default_factory=dict, description="Per-type port defaults"
    )
    policies: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Execution policies (retry, timeout, error handling)",
    )
    aliases: dict[str, str] = Field(
        default_factory=dict,
        description=("User-defined short names for node kinds (maps alias to full module path)"),
    )
    custom_types: dict[str, CustomTypeConfig] = Field(
        default_factory=dict,
        description="Custom sanitized types for output_schema validation",
    )
    services: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Service configurations. Format: {service_name: {class: str, config: dict}}",
    )

    # Schema declarations
    input_schema: dict[str, Any] | None = Field(
        default=None, description="JSON Schema for pipeline inputs"
    )
    output_schema: dict[str, Any] | None = Field(
        default=None, description="JSON Schema for pipeline outputs"
    )
    common_field_mappings: dict[str, Any] | None = Field(
        default=None, description="Reusable field mapping definitions"
    )

    # Execution configuration (optional per-pipeline overrides)
    orchestrator: OrchestratorConfig | None = Field(
        default=None,
        description="Orchestrator execution settings (max_concurrent_nodes, timeout, etc.)",
    )
    limits: DefaultLimits | None = Field(
        default=None,
        description="Resource limits for this pipeline (max_llm_calls, max_cost_usd, etc.)",
    )
    caps: DefaultCaps | None = Field(
        default=None,
        description="Capability boundaries for this pipeline",
    )

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict, description="Pipeline metadata")

    # Node configurations (handled by existing builder)
    nodes: list[dict[str, Any]] = Field(default_factory=list, description="Node specifications")


_pipeline_config_rebuilt = False


def _rebuild_pipeline_config() -> None:
    """Rebuild PipelineConfig to resolve forward references.

    Idempotent — safe to call multiple times (no-op after first).
    This resolves the ``OrchestratorConfig``, ``DefaultLimits``, and
    ``DefaultCaps`` forward references that cannot be imported eagerly due
    to circular imports between ``kernel.domain`` and ``kernel.config``.

    Called automatically from ``kernel/__init__.py`` on first import,
    and also as a safety net from any code that directly imports
    ``PipelineConfig`` without going through ``hexdag.kernel``.
    """
    global _pipeline_config_rebuilt  # noqa: PLW0603
    if _pipeline_config_rebuilt:
        return
    _pipeline_config_rebuilt = True

    from hexdag.kernel.config.models import DefaultCaps, DefaultLimits  # lazy: circular import
    from hexdag.kernel.orchestration.models import OrchestratorConfig  # lazy: circular import

    PipelineConfig.model_rebuild(
        _types_namespace={
            "OrchestratorConfig": OrchestratorConfig,
            "DefaultLimits": DefaultLimits,
            "DefaultCaps": DefaultCaps,
        },
    )
