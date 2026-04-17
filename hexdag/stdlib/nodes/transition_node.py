"""TransitionNode — built-in node for entity state transitions.

Validates and executes a state transition through the pipeline-scoped
``EntityState`` service.  Works in both workflow (DAG step) and agentic
(agent tool call) modes — both paths go through ``EntityState.atransition()``.

YAML usage::

    - kind: transition
      metadata:
        name: accept_load
      spec:
        entity: load
        entity_id: $input.load_id
        to_state: ACCEPTED
        reason: "Smart acceptance criteria met"
"""

from __future__ import annotations

from typing import Any

from hexdag.kernel.domain.dag import NodeSpec
from hexdag.stdlib.nodes.base_node_factory import BaseNodeFactory


class TransitionNode(BaseNodeFactory, yaml_alias="transition"):
    """Factory for entity state transition nodes.

    Creates a node that transitions a business entity to a new state
    via the pipeline-scoped ``EntityState`` service.  The transition is
    validated against the registered state machine, fires transition
    handlers, and emits ``StateTransitionEvent``.
    """

    __aliases__ = ("transition_node",)

    def __call__(
        self,
        name: str,
        *,
        entity: str,
        entity_id: str | None = None,
        to_state: str,
        reason: str | None = None,
        **kwargs: Any,
    ) -> NodeSpec:
        """Build a transition node spec.

        Parameters
        ----------
        name : str
            Node name in the DAG.
        entity : str
            Entity type (must match a registered state machine).
        entity_id : str | None
            Entity ID expression (resolved at runtime from input).
        to_state : str
            Target state for the transition.
        reason : str | None
            Optional reason expression (resolved at runtime).
        **kwargs
            Additional NodeSpec fields (when, on_error, etc.).
        """
        # Capture factory params for the closure
        _entity = entity
        _to_state = to_state
        _reason_expr = reason
        _entity_id_expr = entity_id

        async def _transition(inputs: dict[str, Any], **kw: Any) -> dict[str, Any]:
            from hexdag.kernel.context.execution_context import (
                get_current_node_name,
                get_pipeline_name,
                get_run_id,
                get_services,  # lazy: avoid circular import with kernel
            )
            from hexdag.kernel.orchestration.events.events import (
                TransitionContext,  # lazy: avoid circular import with kernel
            )

            services = get_services()
            if not services or "entity_state" not in services:
                msg = (
                    f"TransitionNode '{name}' requires an 'entity_state' service. "
                    "Declare spec.state_machines or register EntityState as a service."
                )
                raise RuntimeError(msg)

            entity_state = services["entity_state"]

            # Resolve entity_id: prefer input_mapping resolution, fall back to factory param
            resolved_id = inputs.get("entity_id")
            if resolved_id is None:
                resolved_id = _entity_id_expr

            if resolved_id is None:
                msg = f"TransitionNode '{name}': entity_id could not be resolved"
                raise ValueError(msg)

            # Resolve reason from inputs
            resolved_reason = _reason_expr
            if _reason_expr is not None:
                resolved_reason = inputs.get("reason", _reason_expr)

            # Build transition context
            context = TransitionContext(
                run_id=get_run_id() or "",
                pipeline_name=get_pipeline_name() or "",
                node_name=get_current_node_name() or name,
            )

            result: dict[str, Any] = await entity_state.atransition(
                entity_type=_entity,
                entity_id=str(resolved_id),
                to_state=_to_state,
                reason=str(resolved_reason) if resolved_reason else None,
                _context=context,
            )
            return result

        # Extract NodeSpec-level kwargs
        when = kwargs.pop("when", None)
        on_error = kwargs.pop("on_error", None)

        return NodeSpec(
            name=name,
            fn=_transition,
            when=when,
            on_error=on_error,
            factory_class="hexdag.stdlib.nodes.transition_node.TransitionNode",
            factory_params={
                "entity": _entity,
                "entity_id": _entity_id_expr,
                "to_state": _to_state,
                "reason": _reason_expr,
            },
        )
