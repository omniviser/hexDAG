"""System — unified runtime API for ``kind: System`` manifests.

The :class:`System` class is the single object downstream apps hold and
call methods on.  It wraps :class:`SystemBuilder`, :class:`SystemRunner`,
and :class:`LifecycleRunner`, managing shared ports, state machines, and
observers across all pipeline runs.

Usage::

    # DAG mode (ETL-style pipeline chains)
    system = System.from_yaml("etl-system.yaml")
    results = await system.run(input_data)
    await system.close()

    # Lifecycle mode (entity state machines)
    async with System.from_yaml("ticket-system.yaml") as system:
        await system.transition("ticket", "T-1", "INVESTIGATING")
        result = await system.run_process("extract", {"ticket_id": "T-1"})

Library-compatible: no event loop ownership, no daemon threads.
"""

from __future__ import annotations

import contextlib
import pathlib
from typing import TYPE_CHECKING, Any

from hexdag.compiler.system_builder import SystemBuilder
from hexdag.kernel.exceptions import HexDAGError
from hexdag.kernel.lifecycle_runner import LifecycleRunner
from hexdag.kernel.logging import get_logger
from hexdag.kernel.orchestration.orchestrator_factory import OrchestratorFactory
from hexdag.kernel.pipeline_runner import PipelineRunner
from hexdag.kernel.system_runner import SystemRunner

if TYPE_CHECKING:
    from pathlib import Path

    from hexdag.kernel.config.models import HexDAGConfig
    from hexdag.kernel.domain.pipeline_result import PipelineResult
    from hexdag.kernel.domain.system_config import SystemConfig
    from hexdag.kernel.ports.entity_state import EntityState
    from hexdag.kernel.ports.observer_manager import ObserverManager

logger = get_logger(__name__)


class SystemError(HexDAGError):
    """Raised when a System operation is invalid (wrong mode, not started, etc.)."""


class System:
    """Unified runtime API for ``kind: System`` manifests.

    Wraps :class:`SystemConfig` with the appropriate runner (DAG or
    lifecycle), creates shared port instances, and exposes a clean
    public API.

    Parameters
    ----------
    config:
        Compiled system configuration.
    hexdag_config:
        Organisation-wide defaults from ``kind: Config``.
    port_overrides:
        Pre-instantiated port instances to use instead of (or merged
        with) the ports declared in ``spec.ports``.
    observer_manager:
        Optional observer manager for event emission.
    base_path:
        Base directory for resolving relative pipeline paths.
    """

    def __init__(
        self,
        config: SystemConfig,
        *,
        hexdag_config: HexDAGConfig | None = None,
        port_overrides: dict[str, Any] | None = None,
        observer_manager: ObserverManager | None = None,
        base_path: Path | None = None,
    ) -> None:
        """Initialize the System from a compiled configuration."""
        self._config = config
        self._hexdag_config = hexdag_config
        self._observer_manager = observer_manager
        self._base_path = base_path

        # Instantiate shared ports from system config merged with overrides
        self._shared_ports = self._instantiate_system_ports(config, port_overrides)

        # Create the appropriate runner
        self._lifecycle_runner: LifecycleRunner | None = None
        self._system_runner: SystemRunner | None = None

        if config.is_lifecycle:
            self._lifecycle_runner = LifecycleRunner(
                config=hexdag_config,
                port_overrides=self._shared_ports,
                observer_manager=observer_manager,
                base_path=base_path,
            )
        else:
            self._system_runner = SystemRunner(
                config=hexdag_config,
                port_overrides=self._shared_ports,
                observer_manager=observer_manager,
                base_path=base_path,
            )

        self._started = False
        self._process_map = {p.name: p for p in config.processes}

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        hexdag_config: HexDAGConfig | None = None,
        port_overrides: dict[str, Any] | None = None,
        observer_manager: ObserverManager | None = None,
        base_path: Path | None = None,
    ) -> System:
        """Build a System from a ``kind: System`` YAML file.

        Parameters
        ----------
        path:
            Path to the YAML file.
        hexdag_config:
            Organisation-wide defaults.
        port_overrides:
            Pre-instantiated port instances.
        observer_manager:
            Optional observer manager.
        base_path:
            Override base directory for pipeline paths.
        """
        file_path = pathlib.Path(path)
        builder = SystemBuilder(base_path=file_path.parent)
        config = builder.build_from_yaml_file(file_path)
        return cls(
            config,
            hexdag_config=hexdag_config,
            port_overrides=port_overrides,
            observer_manager=observer_manager,
            base_path=base_path or file_path.parent,
        )

    @classmethod
    def from_yaml_string(
        cls,
        yaml_content: str,
        *,
        hexdag_config: HexDAGConfig | None = None,
        port_overrides: dict[str, Any] | None = None,
        observer_manager: ObserverManager | None = None,
        base_path: Path | None = None,
    ) -> System:
        """Build a System from a YAML string.

        Parameters
        ----------
        yaml_content:
            Raw YAML string.
        hexdag_config:
            Organisation-wide defaults.
        port_overrides:
            Pre-instantiated port instances.
        observer_manager:
            Optional observer manager.
        base_path:
            Base directory for resolving pipeline paths.
        """
        builder = SystemBuilder(base_path=base_path)
        config = builder.build_from_yaml_string(yaml_content, base_path=base_path)
        return cls(
            config,
            hexdag_config=hexdag_config,
            port_overrides=port_overrides,
            observer_manager=observer_manager,
            base_path=base_path,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        input_data: dict[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Execute all processes in topological order (DAG mode only).

        Parameters
        ----------
        input_data:
            Initial input data passed to root processes.

        Returns
        -------
        dict[str, dict[str, Any]]
            Results keyed by process name.

        Raises
        ------
        SystemError
            If the system is in lifecycle mode.
        """
        if self._system_runner is None:
            msg = "run() is for DAG-mode systems. Use transition() for lifecycle systems."
            raise SystemError(msg)
        return await self._system_runner.run(self._config, input_data=input_data)

    async def run_process(
        self,
        process_name: str,
        input_data: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """Run a single named process (works in both DAG and lifecycle mode).

        Parameters
        ----------
        process_name:
            Name of the process to run.
        input_data:
            Input data for the pipeline.

        Returns
        -------
        PipelineResult
            Result from the pipeline execution.

        Raises
        ------
        SystemError
            If the process name is not found.
        """
        process_spec = self._process_map.get(process_name)
        if process_spec is None:
            msg = f"Unknown process '{process_name}'. Available: {sorted(self._process_map)}"
            raise SystemError(msg)

        runner = PipelineRunner(
            config=self._hexdag_config,
            port_overrides=self._shared_ports,
            base_path=self._base_path,
        )
        return await runner.run(
            pipeline_path=process_spec.pipeline,
            input_data=input_data,
        )

    async def transition(
        self,
        entity_type: str,
        entity_id: str,
        to_state: str,
        *,
        reason: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Request a state transition (lifecycle mode only).

        Auto-starts the lifecycle runner on first call.

        Parameters
        ----------
        entity_type:
            Entity type (must match a registered state machine).
        entity_id:
            Unique entity identifier.
        to_state:
            Target state.
        reason:
            Optional reason for the transition.
        payload:
            Additional data to pass to the triggered pipeline.

        Returns
        -------
        dict[str, Any]
            Transition result from EntityState.

        Raises
        ------
        SystemError
            If the system is in DAG mode.
        """
        if self._lifecycle_runner is None:
            msg = "transition() is for lifecycle-mode systems. Use run() for DAG systems."
            raise SystemError(msg)

        if not self._started:
            await self._lifecycle_runner.start(self._config)
            self._started = True

        return await self._lifecycle_runner.transition(
            entity_type,
            entity_id,
            to_state,
            reason=reason,
            payload=payload,
        )

    async def close(self) -> None:
        """Shut down the system — stop runners and close port adapters."""
        if self._lifecycle_runner is not None and self._started:
            await self._lifecycle_runner.stop()
            self._started = False

        # Close port adapters that implement aclose/close
        for port in self._shared_ports.values():
            with contextlib.suppress(Exception):
                if hasattr(port, "aclose"):
                    await port.aclose()
                elif hasattr(port, "close"):
                    port.close()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> SystemConfig:
        """Return the compiled system configuration."""
        return self._config

    @property
    def is_lifecycle(self) -> bool:
        """Whether this system operates in lifecycle mode."""
        return self._config.is_lifecycle

    @property
    def process_names(self) -> list[str]:
        """Return ordered list of process names."""
        return self._config.process_names

    @property
    def ports(self) -> dict[str, Any]:
        """Return the shared port instances."""
        return dict(self._shared_ports)

    @property
    def entity_state(self) -> EntityState | None:
        """Return the entity state manager (lifecycle mode only).

        Returns ``None`` if the system is in DAG mode or the lifecycle
        runner has not been started yet.
        """
        if self._lifecycle_runner is not None and self._started:
            return self._lifecycle_runner.entity_state
        return None

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> System:
        """Enter the async context manager."""
        return self

    async def __aexit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """Exit the async context manager and close the system."""
        await self.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _instantiate_system_ports(
        config: SystemConfig,
        overrides: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Instantiate shared port adapters from system config.

        Uses :class:`OrchestratorFactory` to resolve adapter specs,
        then merges with any pre-instantiated overrides.
        """
        ports: dict[str, Any] = {}

        if config.ports:
            factory = OrchestratorFactory()
            ports = factory._instantiate_ports(config.ports)

        if overrides:
            ports.update(overrides)

        return ports
