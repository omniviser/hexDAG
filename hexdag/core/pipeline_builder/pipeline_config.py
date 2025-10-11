"""Pipeline configuration models for YAML-based component configuration.

This module provides Pydantic models for declaring and configuring:
- Global ports (adapters)
- Global policies
- Per-type port defaults (type_ports)
- Per-node port/policy overrides

Note
----
All configuration dictionaries use native YAML dict format.
Formats:
- ports: port_name -> {namespace: str, name: str, params: dict}
- type_ports: node_type -> {port_name -> {namespace: str, name: str, params: dict}}
- policies: policy_name -> {namespace: str, name: str, params: dict}
"""

from typing import Any

from pydantic import BaseModel, Field


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

    # Global configuration
    ports: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Global adapter configurations"
    )
    type_ports: dict[str, dict[str, dict[str, Any]]] = Field(
        default_factory=dict, description="Per-type port defaults"
    )
    policies: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Global policy configurations"
    )

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict, description="Pipeline metadata")

    # Node configurations (handled by existing builder)
    nodes: list[dict[str, Any]] = Field(default_factory=list, description="Node specifications")

    class Config:
        """Pydantic model configuration."""

        extra = "allow"  # Allow additional fields for extensibility
