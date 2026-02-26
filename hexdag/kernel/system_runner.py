"""SystemRunner — execute ``kind: System`` manifests.

Runs each process in topological order (determined by the pipe DAG),
delegating individual process execution to :class:`PipelineRunner`.
Pipe mappings resolve Jinja2 templates to pass outputs from upstream
processes as inputs to downstream processes.

Usage::

    from hexdag.compiler.system_builder import SystemBuilder
    from hexdag.kernel.system_runner import SystemRunner

    builder = SystemBuilder()
    system_config = builder.build_from_yaml_file("system.yaml")

    runner = SystemRunner()
    results = await runner.run(system_config)

Manual mode (one-shot execution) is the initial implementation.
Schedule/continuous/event modes are planned for future phases.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from hexdag.compiler.system_builder import SystemBuilder
from hexdag.kernel.logging import get_logger
from hexdag.kernel.orchestration.events.events import (
    ProcessCompleted,
    ProcessStarted,
    SystemCompleted,
    SystemStarted,
)
from hexdag.kernel.pipeline_runner import PipelineRunner
from hexdag.kernel.utils.node_timer import Timer

if TYPE_CHECKING:
    from pathlib import Path

    from hexdag.kernel.domain.system_config import SystemConfig
    from hexdag.kernel.ports.observer_manager import ObserverManager

logger = get_logger(__name__)

# Simple Jinja2-style template pattern: {{ process_name.field }}
_TEMPLATE_PATTERN = re.compile(r"\{\{\s*(\w+)\.(\w+)\s*\}\}")


class SystemRunner:
    """Execute a :class:`SystemConfig` by running processes in topological order.

    Each process is executed via :class:`PipelineRunner`. Pipe mappings
    resolve ``{{ upstream.field }}`` templates to connect process outputs
    to downstream inputs.

    Parameters
    ----------
    port_overrides:
        Port overrides passed to every :class:`PipelineRunner` instance.
    observer_manager:
        Optional observer manager for emitting system-level events.
    base_path:
        Base directory for resolving relative pipeline paths.
    """

    def __init__(
        self,
        *,
        port_overrides: dict[str, Any] | None = None,
        observer_manager: ObserverManager | None = None,
        base_path: Path | None = None,
    ) -> None:
        self._port_overrides = port_overrides
        self._observer_manager = observer_manager
        self._base_path = base_path

    async def run(
        self,
        system_config: SystemConfig,
        input_data: dict[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Execute all processes in topological order.

        Parameters
        ----------
        system_config:
            Compiled system configuration.
        input_data:
            Initial input data passed to root processes (those with no
            incoming pipes).

        Returns
        -------
        dict[str, dict[str, Any]]
            Results keyed by process name. Each value is the dict of
            node results returned by that process's pipeline.
        """

        system_name = system_config.metadata.get("name", "unnamed")
        execution_order = SystemBuilder.topological_order(system_config)

        system_timer = Timer()

        # Emit SystemStarted
        await self._notify(
            SystemStarted(
                name=system_name,
                total_processes=len(execution_order),
                execution_order=list(execution_order),
            )
        )

        # Build lookup maps
        process_map = {p.name: p for p in system_config.processes}
        pipe_map: dict[str, list[Any]] = {}  # to_process -> list of pipes
        for pipe in system_config.domain_pipes:
            pipe_map.setdefault(pipe.to_process, []).append(pipe)

        # Track results per process
        process_results: dict[str, dict[str, Any]] = {}

        try:
            for idx, process_name in enumerate(execution_order):
                process_spec = process_map[process_name]
                process_timer = Timer()

                await self._notify(
                    ProcessStarted(
                        system_name=system_name,
                        process_name=process_name,
                        index=idx,
                    )
                )

                # Resolve input data for this process from pipe mappings
                process_input = self._resolve_process_input(
                    process_name=process_name,
                    pipe_map=pipe_map,
                    process_results=process_results,
                    initial_input=input_data or {},
                )

                # Create a PipelineRunner for this process
                runner = PipelineRunner(
                    port_overrides=self._port_overrides,
                    base_path=self._base_path,
                )

                try:
                    result = await runner.run(
                        pipeline_path=process_spec.pipeline,
                        input_data=process_input,
                    )
                    process_results[process_name] = result

                    await self._notify(
                        ProcessCompleted(
                            system_name=system_name,
                            process_name=process_name,
                            index=idx,
                            duration_ms=process_timer.duration_ms,
                            status="completed",
                        )
                    )

                    logger.info(
                        "Process '{}' completed in {:.2f}ms ({} node results)",
                        process_name,
                        process_timer.duration_ms,
                        len(result),
                    )

                except Exception as e:
                    await self._notify(
                        ProcessCompleted(
                            system_name=system_name,
                            process_name=process_name,
                            index=idx,
                            duration_ms=process_timer.duration_ms,
                            status="failed",
                            error=str(e),
                        )
                    )

                    await self._notify(
                        SystemCompleted(
                            name=system_name,
                            duration_ms=system_timer.duration_ms,
                            process_results=process_results,
                            status="failed",
                            reason=f"Process '{process_name}' failed: {e}",
                        )
                    )

                    logger.error("Process '{}' failed: {}", process_name, e)
                    raise

        except Exception:
            raise
        else:
            await self._notify(
                SystemCompleted(
                    name=system_name,
                    duration_ms=system_timer.duration_ms,
                    process_results=process_results,
                    status="completed",
                )
            )

            logger.info(
                "System '{}' completed in {:.2f}ms ({} processes)",
                system_name,
                system_timer.duration_ms,
                len(process_results),
            )

        return process_results

    @staticmethod
    def _resolve_process_input(
        process_name: str,
        pipe_map: dict[str, list[Any]],
        process_results: dict[str, dict[str, Any]],
        initial_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve input data for a process from pipe mappings.

        For root processes (no incoming pipes), returns *initial_input*.
        For downstream processes, resolves ``{{ upstream.field }}`` templates
        in pipe mappings against upstream process results.

        Parameters
        ----------
        process_name:
            Target process name.
        pipe_map:
            Map of ``to_process`` -> list of incoming pipes.
        process_results:
            Results from already-executed upstream processes.
        initial_input:
            Input data for root processes.

        Returns
        -------
        dict[str, Any]
            Resolved input data for the process.
        """
        incoming_pipes = pipe_map.get(process_name)
        if not incoming_pipes:
            return dict(initial_input)

        resolved: dict[str, Any] = {}

        for pipe in incoming_pipes:
            for target_field, template in pipe.mapping.items():
                resolved[target_field] = _resolve_template(template, process_results)

        return resolved

    async def _notify(self, event: Any) -> None:
        """Emit an event via the observer manager (if configured)."""
        if self._observer_manager is not None:
            await self._observer_manager.notify(event)


def _resolve_template(template: str, process_results: dict[str, dict[str, Any]]) -> Any:
    """Resolve a ``{{ process.field }}`` template against process results.

    If the template is a simple reference (the entire string is one
    ``{{ x.y }}``), returns the raw value (preserving type).  Otherwise
    performs string interpolation.

    Parameters
    ----------
    template:
        Template string, e.g. ``{{ extract.records }}``.
    process_results:
        Map of process_name -> node_results dict.

    Returns
    -------
    Any
        Resolved value (raw object for simple refs, string for interpolated).
    """
    # Fast path: entire string is a single template reference
    match = _TEMPLATE_PATTERN.fullmatch(template.strip())
    if match:
        process_name, field_name = match.group(1), match.group(2)
        results = process_results.get(process_name, {})
        return results.get(field_name)

    # Slow path: string interpolation for mixed templates
    def _replacer(m: re.Match[str]) -> str:
        process_name, field_name = m.group(1), m.group(2)
        results = process_results.get(process_name, {})
        return str(results.get(field_name, ""))

    return _TEMPLATE_PATTERN.sub(_replacer, template)
