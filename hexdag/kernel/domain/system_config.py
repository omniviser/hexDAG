"""Domain models for ``kind: System`` — multi-process orchestration.

A **System** is an ensemble of named processes (each backed by a Pipeline
YAML) connected by **Pipes** that carry data from one process to the next.

The compiler parses ``kind: System`` YAML into a :class:`SystemConfig` which
the :class:`SystemRunner <hexdag.kernel.system_runner.SystemRunner>` executes
by running each process in topological order determined by the pipe DAG.

YAML example::

    apiVersion: hexdag/v1
    kind: System
    metadata:
      name: etl-system
    spec:
      processes:
        - name: extract
          pipeline: ./pipelines/extract.yaml

        - name: transform
          pipeline: ./pipelines/transform.yaml

        - name: load
          pipeline: ./pipelines/load.yaml

      pipes:
        - from: extract
          to: transform
          mapping:
            records: "{{ extract.records }}"

        - from: transform
          to: load
          mapping:
            results: "{{ transform.results }}"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Frozen dataclass — lightweight, immutable (used inside SystemRunner)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Pipe:
    """A data-flow edge between two processes.

    Attributes
    ----------
    from_process:
        Name of the upstream process.
    to_process:
        Name of the downstream process.
    mapping:
        Field mapping from upstream outputs to downstream inputs.
        Values may contain Jinja2 templates like ``{{ extract.records }}``.
    """

    from_process: str
    to_process: str
    mapping: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pydantic models — used by the compiler for parsing + validation
# ---------------------------------------------------------------------------


class ProcessSpec(BaseModel):
    """Specification for a single process within a System.

    Attributes
    ----------
    name:
        Unique process identifier within the system.
    pipeline:
        Path to the Pipeline YAML file (relative to system manifest),
        or inline pipeline name.
    input_schema:
        Optional JSON-Schema-style dict describing expected inputs.
    output_schema:
        Optional JSON-Schema-style dict describing expected outputs.
    ports:
        Per-process port overrides (merged over global ports).
    policies:
        Per-process policy overrides.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Unique process name")
    pipeline: str = Field(description="Path to Pipeline YAML or pipeline name")
    input_schema: dict[str, Any] | None = Field(
        default=None, description="JSON-Schema for process inputs"
    )
    output_schema: dict[str, Any] | None = Field(
        default=None, description="JSON-Schema for process outputs"
    )
    ports: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Per-process port overrides"
    )
    policies: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Per-process policy overrides"
    )


class PipeSpec(BaseModel):
    """Pydantic representation of a :class:`Pipe` for YAML parsing.

    Uses ``from`` / ``to`` field names (aliased because ``from`` is a
    Python keyword).
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    from_process: str = Field(alias="from", description="Upstream process name")
    to_process: str = Field(alias="to", description="Downstream process name")
    mapping: dict[str, str] = Field(
        default_factory=dict,
        description="Field mapping (values may use Jinja2 templates)",
    )

    def to_domain(self) -> Pipe:
        """Convert to the frozen dataclass used at runtime."""
        return Pipe(
            from_process=self.from_process,
            to_process=self.to_process,
            mapping=dict(self.mapping),
        )


class SystemConfig(BaseModel):
    """Compiled configuration for a ``kind: System`` manifest.

    Produced by :class:`SystemBuilder <hexdag.compiler.system_builder.SystemBuilder>`
    and consumed by :class:`SystemRunner <hexdag.kernel.system_runner.SystemRunner>`.
    """

    model_config = ConfigDict(extra="allow")

    metadata: dict[str, Any] = Field(default_factory=dict, description="System metadata")
    processes: list[ProcessSpec] = Field(default_factory=list, description="Process definitions")
    pipes: list[PipeSpec] = Field(
        default_factory=list, description="Data-flow edges between processes"
    )
    ports: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Global port configurations (shared across processes)"
    )
    policies: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Global execution policies"
    )
    services: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Global service configurations"
    )

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def process_names(self) -> list[str]:
        """Return ordered list of process names."""
        return [p.name for p in self.processes]

    @property
    def domain_pipes(self) -> list[Pipe]:
        """Return pipes as frozen dataclass instances."""
        return [p.to_domain() for p in self.pipes]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _validate_pipe_references(self) -> SystemConfig:
        """Ensure all pipe endpoints reference declared processes."""
        names = set(self.process_names)
        for pipe in self.pipes:
            if pipe.from_process not in names:
                msg = (
                    f"Pipe references unknown process '{pipe.from_process}'. "
                    f"Declared processes: {sorted(names)}"
                )
                raise ValueError(msg)
            if pipe.to_process not in names:
                msg = (
                    f"Pipe references unknown process '{pipe.to_process}'. "
                    f"Declared processes: {sorted(names)}"
                )
                raise ValueError(msg)
            if pipe.from_process == pipe.to_process:
                msg = f"Pipe cannot connect process '{pipe.from_process}' to itself"
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_unique_process_names(self) -> SystemConfig:
        """Ensure process names are unique."""
        seen: set[str] = set()
        for p in self.processes:
            if p.name in seen:
                msg = f"Duplicate process name: '{p.name}'"
                raise ValueError(msg)
            seen.add(p.name)
        return self
