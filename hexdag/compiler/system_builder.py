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
                "state_machines": spec.get("state_machines", {}),
                "states": spec.get("states", {}),
                "on_transition": spec.get("on_transition", {}),
                "observers": spec.get("observers", []),
                "memory": spec.get("memory", {}),
                "gc": spec.get("gc", {}),
            })
        except Exception as e:
            raise SystemBuildError(f"System validation error: {e}") from e

        # 5. Validate pipeline paths exist
        self._validate_pipeline_paths(system_config, resolved_base)

        # 6. Validate pipe DAG is acyclic (only for non-lifecycle systems)
        if not system_config.is_lifecycle:
            self._validate_no_cycles(system_config)

        # 7. Validate lifecycle constraints (if state machines declared)
        if system_config.is_lifecycle:
            self._validate_lifecycle(system_config)

        system_name = metadata.get("name", "unnamed")
        mode = "lifecycle" if system_config.is_lifecycle else "dag"
        logger.info(
            "Built system '{}' ({}): {} processes, {} pipes, {} state machines",
            system_name,
            mode,
            len(system_config.processes),
            len(system_config.pipes),
            len(system_config.state_machines),
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
    def _validate_lifecycle(config: SystemConfig) -> None:
        """Validate lifecycle-mode constraints.

        Ensures that ``states`` and ``on_transition`` reference valid
        process names and valid state machine states.

        Raises
        ------
        SystemBuildError
            If lifecycle configuration is invalid.
        """
        process_names = set(config.process_names)

        # Collect all valid states from state machines
        all_states: set[str] = set()
        for sm_spec in config.state_machines.values():
            transitions = sm_spec.get("transitions", {})
            for from_state, targets in transitions.items():
                all_states.add(from_state)
                if isinstance(targets, list):
                    for t in targets:
                        if isinstance(t, str):
                            all_states.add(t)
                        elif isinstance(t, dict) and "to" in t:
                            all_states.add(t["to"])
            initial = sm_spec.get("initial")
            if initial:
                all_states.add(initial)

        # Validate spec.states references
        for state_name, state_spec in config.states.items():
            if state_name not in all_states:
                raise SystemBuildError(
                    f"spec.states references unknown state '{state_name}'. "
                    f"Valid states: {sorted(all_states)}"
                )
            on_enter = state_spec.get("on_enter")
            if on_enter and on_enter not in process_names:
                raise SystemBuildError(
                    f"State '{state_name}' references unknown process "
                    f"'{on_enter}'. Declared processes: {sorted(process_names)}"
                )

        # Validate spec.on_transition references
        for transition_key, transition_spec in config.on_transition.items():
            parts = [p.strip() for p in transition_key.split("->")]
            if len(parts) != 2:  # noqa: PLR2004
                raise SystemBuildError(
                    f"Invalid on_transition key '{transition_key}'. "
                    "Expected format: 'FROM_STATE -> TO_STATE'"
                )
            from_state, to_state = parts
            if from_state not in all_states:
                raise SystemBuildError(
                    f"on_transition '{transition_key}' references unknown state '{from_state}'"
                )
            if to_state not in all_states:
                raise SystemBuildError(
                    f"on_transition '{transition_key}' references unknown state '{to_state}'"
                )
            process_name = transition_spec.get("process")
            if process_name and process_name not in process_names:
                raise SystemBuildError(
                    f"on_transition '{transition_key}' references unknown "
                    f"process '{process_name}'. "
                    f"Declared processes: {sorted(process_names)}"
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
