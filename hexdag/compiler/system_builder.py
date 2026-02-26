"""SystemBuilder — compile ``kind: System`` YAML into :class:`SystemConfig`.

Mirrors :class:`YamlPipelineBuilder` but for system-level manifests.
Parses the YAML, validates process/pipe declarations, and produces a
:class:`SystemConfig` that :class:`SystemRunner` can execute.

Usage::

    builder = SystemBuilder()
    system_config = builder.build_from_yaml_file("system.yaml")

Or from a string::

    system_config = builder.build_from_yaml_string(yaml_content)
"""

from __future__ import annotations

from pathlib import Path

import yaml

from hexdag.kernel.domain.system_config import SystemConfig
from hexdag.kernel.exceptions import HexDAGError
from hexdag.kernel.logging import get_logger

logger = get_logger(__name__)


class SystemBuildError(HexDAGError):
    """Raised when a ``kind: System`` manifest cannot be compiled."""


class SystemBuilder:
    """Compile ``kind: System`` YAML manifests into :class:`SystemConfig`.

    Parameters
    ----------
    base_path:
        Base directory for resolving relative pipeline paths.
        Defaults to the current working directory.
    """

    def __init__(self, base_path: Path | None = None) -> None:
        self._base_path = base_path or Path.cwd()

    def build_from_yaml_file(self, path: str | Path) -> SystemConfig:
        """Parse a ``kind: System`` YAML file.

        Parameters
        ----------
        path:
            Path to the YAML file.

        Returns
        -------
        SystemConfig
            Compiled system configuration.

        Raises
        ------
        SystemBuildError
            If the file is missing or malformed.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise SystemBuildError(f"System manifest not found: {file_path}")

        yaml_content = file_path.read_text(encoding="utf-8")
        return self.build_from_yaml_string(yaml_content, base_path=file_path.parent)

    def build_from_yaml_string(
        self,
        yaml_content: str,
        *,
        base_path: Path | None = None,
    ) -> SystemConfig:
        """Parse a ``kind: System`` YAML string.

        Parameters
        ----------
        yaml_content:
            Raw YAML string.
        base_path:
            Override base path for relative pipeline paths.

        Returns
        -------
        SystemConfig
            Compiled system configuration.

        Raises
        ------
        SystemBuildError
            If the YAML is malformed or fails validation.
        """
        resolved_base = base_path or self._base_path

        # 1. Parse YAML
        try:
            doc = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            raise SystemBuildError(f"YAML parse error: {e}") from e

        if not isinstance(doc, dict):
            raise SystemBuildError("System manifest must be a YAML mapping")

        # 2. Validate kind
        kind = doc.get("kind")
        if kind != "System":
            raise SystemBuildError(f"Expected kind: System, got kind: {kind}")

        # 3. Extract sections
        metadata = doc.get("metadata", {})
        spec = doc.get("spec", {})
        if not spec:
            raise SystemBuildError("System manifest missing 'spec' section")

        # 4. Build SystemConfig via Pydantic validation
        try:
            system_config = SystemConfig.model_validate({
                "metadata": metadata,
                "processes": spec.get("processes", []),
                "pipes": spec.get("pipes", []),
                "ports": spec.get("ports", {}),
                "policies": spec.get("policies", {}),
                "services": spec.get("services", {}),
            })
        except Exception as e:
            raise SystemBuildError(f"System validation error: {e}") from e

        # 5. Validate pipeline paths exist
        self._validate_pipeline_paths(system_config, resolved_base)

        # 6. Validate pipe DAG is acyclic
        self._validate_no_cycles(system_config)

        system_name = metadata.get("name", "unnamed")
        logger.info(
            "Built system '{}': {} processes, {} pipes",
            system_name,
            len(system_config.processes),
            len(system_config.pipes),
        )

        return system_config

    @staticmethod
    def _validate_pipeline_paths(config: SystemConfig, base_path: Path) -> None:
        """Verify that pipeline YAML files exist for all processes.

        Raises
        ------
        SystemBuildError
            If any referenced pipeline file is missing.
        """
        for process in config.processes:
            pipeline_path = base_path / process.pipeline
            if not pipeline_path.exists():
                raise SystemBuildError(
                    f"Process '{process.name}' references missing pipeline: {pipeline_path}"
                )

    @staticmethod
    def _validate_no_cycles(config: SystemConfig) -> None:
        """Verify that the pipe DAG contains no cycles.

        Uses Kahn's algorithm (topological sort with in-degree tracking).

        Raises
        ------
        SystemBuildError
            If a cycle is detected.
        """
        if not config.pipes:
            return

        # Build adjacency list + in-degree map
        names = set(config.process_names)
        adj: dict[str, list[str]] = {n: [] for n in names}
        in_degree: dict[str, int] = dict.fromkeys(names, 0)

        for pipe in config.pipes:
            adj[pipe.from_process].append(pipe.to_process)
            in_degree[pipe.to_process] += 1

        # Kahn's algorithm
        queue = [n for n in names if in_degree[n] == 0]
        visited = 0

        while queue:
            node = queue.pop(0)
            visited += 1
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if visited != len(names):
            raise SystemBuildError(
                "Cycle detected in pipe DAG. Processes must form a directed acyclic graph."
            )

    @staticmethod
    def topological_order(config: SystemConfig) -> list[str]:
        """Compute topological execution order from the pipe DAG.

        Processes with no incoming pipes run first.  The returned order
        respects all pipe dependencies.

        Parameters
        ----------
        config:
            Validated system configuration.

        Returns
        -------
        list[str]
            Process names in execution order.
        """
        names = list(config.process_names)

        if not config.pipes:
            return names  # No pipes → original declaration order

        adj: dict[str, list[str]] = {n: [] for n in names}
        in_degree: dict[str, int] = dict.fromkeys(names, 0)

        for pipe in config.pipes:
            adj[pipe.from_process].append(pipe.to_process)
            in_degree[pipe.to_process] += 1

        # Kahn's algorithm — stable sort (use sorted queue for determinism)
        queue = sorted(n for n in names if in_degree[n] == 0)
        order: list[str] = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for neighbor in sorted(adj[node]):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return order
