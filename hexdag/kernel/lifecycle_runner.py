"""LifecycleRunner — event-driven execution for lifecycle-aware Systems.

When a ``kind: System`` manifest declares ``spec.state_machines``, the
:class:`SystemBuilder` routes execution to this runner instead of the
default DAG-mode :class:`SystemRunner`.

The runner maintains per-entity inboxes, validates transitions against
registered state machines, fires transition handlers, emits events, and
spawns the appropriate pipeline process for each state entry.

Usage::

    from hexdag.compiler.system_builder import SystemBuilder
    from hexdag.kernel.lifecycle_runner import LifecycleRunner

    builder = SystemBuilder()
    system_config = builder.build_from_yaml_file("system.yaml")

    runner = LifecycleRunner()
    await runner.start(system_config)

    # Trigger a transition (from API, webhook, timer, etc.)
    result = await runner.transition("ticket", "T-123", "INVESTIGATING")

    # Graceful shutdown
    await runner.stop()
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from hexdag.kernel.domain.entity_state import StateMachineConfig
from hexdag.kernel.exceptions import HexDAGError, InvalidTransitionError
from hexdag.kernel.logging import get_logger
from hexdag.kernel.orchestration.events.events import (
    EntityGarbageCollected,
    TransitionContext,
)
from hexdag.kernel.pipeline_runner import PipelineRunner
from hexdag.kernel.utils.node_timer import Timer
from hexdag.stdlib.lib.entity_state import EntityState

if TYPE_CHECKING:
    from pathlib import Path

    from hexdag.kernel.config.models import HexDAGConfig
    from hexdag.kernel.domain.system_config import SystemConfig
    from hexdag.kernel.ports.observer_manager import ObserverManager

logger = get_logger(__name__)

# Default limits
_DEFAULT_MAX_CASCADE_DEPTH = 10


class LifecycleError(HexDAGError):
    """Raised when a lifecycle operation fails."""


class CascadeDepthExceeded(LifecycleError):
    """Raised when transition cascading exceeds the configured depth limit."""


@dataclass(slots=True)
class TransitionRequest:
    """A request to transition an entity to a new state."""

    entity_type: str
    entity_id: str
    to_state: str
    reason: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    cascade_depth: int = 0


@dataclass(slots=True)
class EntityRecord:
    """Tracks an active entity in the lifecycle runner."""

    entity_type: str
    entity_id: str
    created_at: float = field(default_factory=time.time)
    transition_count: int = 0


class LifecycleRunner:
    """Event-driven runner for lifecycle-aware Systems.

    Maintains per-entity inboxes for sequential transition processing,
    validates against registered state machines, and spawns pipeline
    processes on state entry.

    Parameters
    ----------
    config:
        Organisation-wide defaults from ``kind: Config``.
    port_overrides:
        Port overrides passed to PipelineRunner.
    observer_manager:
        Optional observer manager for event emission.
    base_path:
        Base directory for resolving pipeline paths.
    max_cascade_depth:
        Maximum depth for recursive transitions (default: 10).
    """

    def __init__(
        self,
        *,
        config: HexDAGConfig | None = None,
        port_overrides: dict[str, Any] | None = None,
        observer_manager: ObserverManager | None = None,
        base_path: Path | None = None,
        max_cascade_depth: int = _DEFAULT_MAX_CASCADE_DEPTH,
    ) -> None:
        self._config = config
        self._port_overrides = port_overrides
        self._observer_manager = observer_manager
        self._base_path = base_path
        self._max_cascade_depth = max_cascade_depth

        # Populated on start()
        self._system_config: SystemConfig | None = None
        self._entity_state: EntityState | None = None
        self._entities: dict[tuple[str, str], EntityRecord] = {}
        self._running = False
        self._draining = False

        # Lookup tables built from system config
        self._state_to_process: dict[str, str] = {}  # STATE -> process_name
        self._transition_to_process: dict[tuple[str, str], str] = {}
        self._terminal_states: set[str] = set()
        self._state_requires: dict[str, list[str]] = {}  # STATE -> required fields
        self._process_map: dict[str, Any] = {}
        self._guards: dict[tuple[str, str], str] = {}  # (from, to) -> expression

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, system_config: SystemConfig) -> None:
        """Initialize the lifecycle runner from a system config.

        Parses state machines, builds lookup tables, and prepares
        the runner for accepting transitions.
        """
        if self._running:
            msg = "LifecycleRunner is already started"
            raise LifecycleError(msg)

        self._system_config = system_config
        self._running = True
        self._draining = False

        # Build EntityState from state machines
        self._entity_state = EntityState()
        for entity_type, sm_spec in system_config.state_machines.items():
            transitions_raw = sm_spec.get("transitions", {})
            transitions: dict[str, set[str]] = {}
            for from_state, targets in transitions_raw.items():
                target_set: set[str] = set()
                if isinstance(targets, list):
                    for t in targets:
                        if isinstance(t, str):
                            target_set.add(t)
                        elif isinstance(t, dict) and "to" in t:
                            target_set.add(t["to"])
                            # Extract guard if present
                            guard = t.get("guard")
                            if guard:
                                self._guards[(from_state, t["to"])] = guard
                transitions[from_state] = target_set

            all_states = set(transitions.keys())
            for ts in transitions.values():
                all_states |= ts
            initial = sm_spec.get("initial", "")
            if initial:
                all_states.add(initial)

            config = StateMachineConfig(
                entity_type=entity_type,
                states=all_states,
                initial_state=initial,
                transitions=transitions,
            )
            self._entity_state.register_machine(config)

            # Register handler if declared
            handlers = sm_spec.get("handlers", {})
            handler_path = handlers.get("on_transition")
            if handler_path:
                from hexdag.kernel.resolver import resolve

                handler_cls = resolve(handler_path)
                handler = handler_cls() if isinstance(handler_cls, type) else handler_cls
                self._entity_state.register_handler(entity_type, handler)

        # Build state -> process lookup + data contracts
        for state_name, state_spec in system_config.states.items():
            on_enter = state_spec.get("on_enter")
            if on_enter:
                self._state_to_process[state_name] = on_enter
            if state_spec.get("terminal"):
                self._terminal_states.add(state_name)
            requires = state_spec.get("requires")
            if requires and isinstance(requires, list):
                self._state_requires[state_name] = requires

        # Build transition-specific process lookup
        for transition_key, transition_spec in system_config.on_transition.items():
            parts = [p.strip() for p in transition_key.split("->")]
            if len(parts) == 2:  # noqa: PLR2004
                from_state, to_state = parts
                process_name = transition_spec.get("process")
                if process_name:
                    self._transition_to_process[(from_state, to_state)] = process_name

        # Build process map
        self._process_map = {p.name: p for p in system_config.processes}

        system_name = system_config.metadata.get("name", "unnamed")
        logger.info(
            "LifecycleRunner started for system '{}' "
            "({} state machines, {} state mappings, {} terminal states)",
            system_name,
            len(system_config.state_machines),
            len(self._state_to_process),
            len(self._terminal_states),
        )

    async def stop(self, timeout: float = 30.0) -> None:
        """Graceful shutdown — drain in-flight transitions, then stop.

        Parameters
        ----------
        timeout:
            Maximum seconds to wait for in-flight transitions to complete.
        """
        if not self._running:
            return

        self._draining = True
        logger.info("LifecycleRunner draining (timeout={}s)...", timeout)

        # v1: single-process, synchronous execution — no in-flight tasks to drain.
        # Future: per-entity async tasks + inbox queues for concurrent processing.

        self._running = False
        self._draining = False
        self._entities.clear()
        logger.info("LifecycleRunner stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def transition(
        self,
        entity_type: str,
        entity_id: str,
        to_state: str,
        reason: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Request a state transition for an entity.

        If the entity is new (not yet registered), it is auto-registered
        with the state machine's initial state.

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
        LifecycleError
            If the runner is not started or is draining.
        """
        if not self._running or self._entity_state is None:
            msg = "LifecycleRunner is not started"
            raise LifecycleError(msg)

        if self._draining:
            msg = "LifecycleRunner is shutting down; no new transitions accepted"
            raise LifecycleError(msg)

        # Auto-register entity if not yet tracked
        key = (entity_type, entity_id)
        if key not in self._entities:
            await self._register_entity(entity_type, entity_id)

        # Execute transition directly (single-process v1: no inbox queue needed)
        return await self._execute_transition(
            TransitionRequest(
                entity_type=entity_type,
                entity_id=entity_id,
                to_state=to_state,
                reason=reason,
                payload=payload or {},
            )
        )

    # ------------------------------------------------------------------
    # Internal transition processing
    # ------------------------------------------------------------------

    async def _register_entity(
        self,
        entity_type: str,
        entity_id: str,
        initial_state: str | None = None,
    ) -> None:
        """Register an entity with the lifecycle runner."""
        if self._entity_state is None:
            return

        key = (entity_type, entity_id)
        if key in self._entities:
            return

        # Validate entity_type is a known state machine
        if entity_type not in self._entity_state:
            msg = f"Unknown entity type '{entity_type}'. Registered: {sorted(self._entity_state)}"
            raise LifecycleError(msg)

        await self._entity_state.aregister_entity(
            entity_type,
            entity_id,
            initial_state=initial_state,
        )

        self._entities[key] = EntityRecord(
            entity_type=entity_type,
            entity_id=entity_id,
        )

    async def _execute_transition(
        self,
        request: TransitionRequest,
    ) -> dict[str, Any]:
        """Execute a single transition: validate, commit, spawn pipeline."""
        if self._entity_state is None or self._system_config is None:
            msg = "LifecycleRunner not initialized"
            raise LifecycleError(msg)

        # Check cascade depth
        if request.cascade_depth > self._max_cascade_depth:
            msg = (
                f"Cascade depth exceeded ({request.cascade_depth}) for "
                f"{request.entity_type}:{request.entity_id} -> {request.to_state}. "
                f"Max allowed: {self._max_cascade_depth}"
            )
            raise CascadeDepthExceeded(msg)

        # Evaluate guard (if any)
        current_state = await self._entity_state.aget_state(
            request.entity_type,
            request.entity_id,
        )
        current = current_state["state"] if current_state else None
        if current:
            guard_expr = self._guards.get((current, request.to_state))
            if guard_expr and not self._evaluate_guard(guard_expr, request):
                msg = (
                    f"Guard blocked transition "
                    f"{request.entity_type}:{request.entity_id} "
                    f"{current} -> {request.to_state}: {guard_expr}"
                )
                raise InvalidTransitionError(msg)

        # Validate state data contract (requires)
        required_fields = self._state_requires.get(request.to_state)
        if required_fields:
            missing = [f for f in required_fields if request.payload.get(f) is None]
            if missing:
                msg = (
                    f"State '{request.to_state}' requires fields "
                    f"{missing} but they are missing from the transition payload"
                )
                raise LifecycleError(msg)

        # Build transition context
        system_name = self._system_config.metadata.get("name", "")
        context = TransitionContext(
            run_id="",
            pipeline_name=system_name,
            node_name="lifecycle_runner",
        )

        # Execute transition (validates, fires handler, emits event)
        result = await self._entity_state.atransition(
            entity_type=request.entity_type,
            entity_id=request.entity_id,
            to_state=request.to_state,
            reason=request.reason,
            _context=context,
        )

        # Track
        key = (request.entity_type, request.entity_id)
        record = self._entities.get(key)
        if record:
            record.transition_count += 1

        from_state = result.get("from_state", "")

        # Determine which process to spawn
        process_name = self._resolve_process(from_state, request.to_state)
        if process_name:
            await self._spawn_process(process_name, request, from_state)

        # Check terminal state → basic GC
        if request.to_state in self._terminal_states:
            await self._gc_entity(key)

        return result

    def _resolve_process(self, from_state: str, to_state: str) -> str | None:
        """Determine which process to run for a transition.

        Resolution order:
        1. Transition-specific process (on_transition: FROM -> TO)
        2. State on_enter process (states: TO_STATE.on_enter)
        """
        # Check transition-specific override first
        process = self._transition_to_process.get((from_state, to_state))
        if process:
            return process

        # Fall back to state on_enter
        return self._state_to_process.get(to_state)

    async def _spawn_process(
        self,
        process_name: str,
        request: TransitionRequest,
        from_state: str,
    ) -> None:
        """Spawn a pipeline process for a state transition."""
        process_spec = self._process_map.get(process_name)
        if not process_spec:
            logger.warning(
                "Process '{}' not found in system config",
                process_name,
            )
            return

        # Build input data for the pipeline
        input_data = {
            "entity_type": request.entity_type,
            "entity_id": request.entity_id,
            "from_state": from_state,
            "to_state": request.to_state,
            "reason": request.reason,
            **request.payload,
        }

        runner = PipelineRunner(
            config=self._config,
            port_overrides=self._port_overrides,
            base_path=self._base_path,
        )

        try:
            timer = Timer()
            await runner.run(
                pipeline_path=process_spec.pipeline,
                input_data=input_data,
            )
            logger.info(
                "Process '{}' completed for {}:{} ({} -> {}) in {:.0f}ms",
                process_name,
                request.entity_type,
                request.entity_id,
                from_state,
                request.to_state,
                timer.duration_ms,
            )
        except Exception:
            logger.exception(
                "Process '{}' failed for {}:{} ({} -> {})",
                process_name,
                request.entity_type,
                request.entity_id,
                from_state,
                request.to_state,
            )
            raise

    def _evaluate_guard(
        self,
        expression: str,
        request: TransitionRequest,
    ) -> bool:
        """Evaluate a guard expression against the transition request."""
        try:
            from hexdag.kernel.expression_parser import (
                compile_expression,  # lazy: avoid import cost when guards not used
            )

            predicate = compile_expression(expression)
            # Payload is spread first so framework fields take precedence
            data_context = {
                **request.payload,
                "input": request.payload,
                "entity_type": request.entity_type,
                "entity_id": request.entity_id,
                "to_state": request.to_state,
                "reason": request.reason,
            }
            result = predicate(data_context, {})
            return bool(result)
        except Exception:
            logger.warning(
                "Guard expression failed for {}:{}: {}",
                request.entity_type,
                request.entity_id,
                expression,
            )
            return False

    # ------------------------------------------------------------------
    # Basic GC (v1: cleanup on terminal state)
    # ------------------------------------------------------------------

    async def _gc_entity(self, key: tuple[str, str]) -> None:
        """Clean up an entity that reached a terminal state."""
        record = self._entities.pop(key, None)
        if record is None:
            return

        entity_type, entity_id = key

        # Get final state info before cleanup
        state_info = None
        if self._entity_state:
            state_info = await self._entity_state.aget_state(entity_type, entity_id)
        final_state = state_info["state"] if state_info else "unknown"

        # Clean up EntityState in-memory data to prevent memory leaks
        if self._entity_state:
            self._entity_state._states.pop(key, None)
            self._entity_state._history.pop(key, None)

        # Calculate lifetime
        lifetime_ms = (time.time() - record.created_at) * 1000

        # Emit GC event
        await self._notify(
            EntityGarbageCollected(
                entity_type=entity_type,
                entity_id=entity_id,
                final_state=final_state,
                lifetime_ms=lifetime_ms,
                transition_count=record.transition_count,
            )
        )

        logger.debug(
            "GC: {}:{} collected (state={}, transitions={}, lifetime={:.0f}ms)",
            entity_type,
            entity_id,
            final_state,
            record.transition_count,
            lifetime_ms,
        )

    async def _notify(self, event: Any) -> None:
        """Emit an event via the observer manager (if configured)."""
        if self._observer_manager is not None:
            with contextlib.suppress(Exception):
                await self._observer_manager.notify(event)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """Whether the runner is currently active."""
        return self._running

    @property
    def is_draining(self) -> bool:
        """Whether the runner is in shutdown drain mode."""
        return self._draining

    @property
    def active_entities(self) -> int:
        """Number of entities currently tracked."""
        return len(self._entities)

    @property
    def entity_state(self) -> EntityState | None:
        """The lifecycle's EntityState service (for external access)."""
        return self._entity_state
