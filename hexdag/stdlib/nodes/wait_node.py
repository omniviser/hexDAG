"""Wait node — suspends pipeline execution until an external event arrives.

This node enables async human-in-the-loop patterns: a pipeline can send
an email, then *park itself* waiting for the reply.  When the reply
arrives (hours or days later), the pipeline resumes with all context
preserved.

YAML usage::

    - kind: wait_node
      metadata:
        name: await_reply
      spec:
        event_key: "email_reply:{{$input.conversation_id}}"
        timeout: 7d
        on_timeout: timeout_handler

On suspend, the node returns a ``Suspended`` signal.  On resume
(via ``PipelineRunner.resume_with_event``), the external event data
becomes this node's output — downstream nodes see it through normal
dependency resolution.
"""

from __future__ import annotations

import re
from typing import Any

from hexdag.kernel.domain.dag import NodeSpec
from hexdag.kernel.logging import get_logger
from hexdag.kernel.orchestration.suspension import Suspended

from .base_node_factory import BaseNodeFactory

logger = get_logger(__name__)

# Duration string pattern: "7d", "24h", "30m", "3600s", "1d12h"
_DURATION_UNITS = {"d": 86400, "h": 3600, "m": 60, "s": 1}
_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([dhms])", re.IGNORECASE)


def _parse_duration(value: str | float | int | None) -> float | None:
    """Parse a duration string into seconds.

    Supports formats: ``"7d"``, ``"24h"``, ``"30m"``, ``"1d12h"``,
    or a raw number (seconds).  Returns ``None`` for ``None`` input.

    Examples
    --------
    >>> _parse_duration("7d")
    604800.0
    >>> _parse_duration("1d12h")
    129600.0
    >>> _parse_duration(3600)
    3600.0
    >>> _parse_duration(None) is None
    True
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    # Try parsing as duration string
    matches = _DURATION_RE.findall(str(value))
    if matches:
        return sum(float(n) * _DURATION_UNITS[u.lower()] for n, u in matches)
    # Fall back to float
    return float(value)


class WaitNode(BaseNodeFactory, yaml_alias="wait_node"):
    """Wait node factory — suspends execution until an external event.

    The node returns a ``Suspended`` signal that the orchestrator
    catches, saves a checkpoint with wait metadata, and returns a
    suspended ``PipelineResult``.

    When the external event arrives, ``PipelineRunner.resume_with_event()``
    injects the event data as this node's output and re-runs remaining
    nodes with full context preserved.

    Parameters (YAML ``spec``)
    --------------------------
    event_key : str
        Correlation key for the external event.  Supports ``{{}}``
        template syntax for dynamic keys.
    timeout : str | float | None
        How long to wait.  Accepts duration strings (``"7d"``,
        ``"24h"``, ``"30m"``) or raw seconds.  ``None`` = no timeout.
    on_timeout : str | None
        Node name to execute if the wait times out.
    setup_fn : str | None
        Optional module path to an async function called before
        suspending.  Its return value is stored in the checkpoint
        as ``setup_result``.

    Examples
    --------
    YAML pipeline::

        - kind: function_node
          metadata:
            name: send_email
          spec:
            fn: myapp.send_counter_email

        - kind: wait_node
          metadata:
            name: await_reply
          spec:
            event_key: "email_reply:{{$input.conversation_id}}"
            timeout: 7d

        - kind: llm_node
          metadata:
            name: analyze_reply
          spec:
            human_message: "Carrier reply: {{await_reply.body}}"
    """

    _hexdag_icon = "Clock"
    _hexdag_color = "#8b5cf6"  # violet-500

    def __call__(
        self,
        name: str,
        event_key: str,
        timeout: str | float | None = None,
        on_timeout: str | None = None,
        setup_fn: str | None = None,
        deps: list[str] | None = None,
        **kwargs: Any,
    ) -> NodeSpec:
        """Create a NodeSpec for a wait node.

        Parameters
        ----------
        name : str
            Node name (must be unique within the pipeline).
        event_key : str
            Correlation key template (e.g. ``"email_reply:{{$input.conversation_id}}"``).
        timeout : str | float | None
            Wait duration.  Accepts ``"7d"``, ``"24h"``, ``3600``, or ``None``.
        on_timeout : str | None
            Node name to route to on timeout.
        setup_fn : str | None
            Module path to async setup function.
        deps : list[str] | None
            Dependency node names.
        **kwargs : Any
            Additional NodeSpec parameters (when, critical, etc.)

        Returns
        -------
        NodeSpec
            Node specification that returns ``Suspended`` on execution.
        """
        timeout_seconds = _parse_duration(timeout)

        # Resolve setup function if provided
        resolved_setup_fn = None
        if setup_fn:
            from hexdag.kernel.resolver import resolve  # lazy: avoid circular import

            resolved_setup_fn = resolve(setup_fn)

        # Capture in closure
        _event_key = event_key
        _timeout_seconds = timeout_seconds
        _on_timeout = on_timeout
        _setup_fn = resolved_setup_fn

        async def _wait_fn(inputs: dict[str, Any], **_kwargs: Any) -> Suspended:
            """Execute setup (if any) and return Suspended signal."""
            # Resolve event_key template
            resolved_key = _event_key
            if "{{" in _event_key:
                from hexdag.kernel.orchestration.prompt.template import (
                    PromptTemplate,  # lazy: deferred to avoid import in hot path
                )

                tpl = PromptTemplate(_event_key)
                resolved_key = tpl.render(**inputs) if isinstance(inputs, dict) else tpl.render()

            setup_result = None
            if _setup_fn is not None:
                import asyncio  # lazy: only needed when setup_fn is provided

                if asyncio.iscoroutinefunction(_setup_fn):
                    setup_result = await _setup_fn(inputs)
                else:
                    setup_result = _setup_fn(inputs)

            return Suspended(
                event_key=resolved_key,
                timeout_seconds=_timeout_seconds,
                setup_result=setup_result,
                metadata={"on_timeout": _on_timeout} if _on_timeout else {},
            )

        return NodeSpec(
            name=name,
            fn=_wait_fn,
            deps=frozenset(deps or []),
            params=kwargs.get("params", {}),
            literals=kwargs.get("literals", {}),
            timeout=kwargs.get("timeout_node"),
            when=kwargs.get("when"),
            on_error=on_timeout,  # on_timeout maps to on_error for orchestrator routing
            critical=kwargs.get("critical", False),
            required_inputs=tuple(kwargs.get("required_inputs", ())),
            factory_class=f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            factory_params={"event_key": event_key, "timeout": timeout, "on_timeout": on_timeout},
        )
