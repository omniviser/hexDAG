"""Port preparation — middleware stacking for observability and control.

The ``prepare_ports()`` function is the single entry point used by the
orchestrator.  It stacks middleware in two phases:

1. **User middleware** — declared in YAML via ``spec.ports.<name>.middleware``
   or ``kind: Middleware`` manifests.  Applied inner-to-outer (first item wraps
   the adapter directly).

2. **Auto middleware** — always applied outermost by the framework:

   - LLM ports (``SupportsGeneration``):
     a. ``StructuredOutputFallback`` — if adapter lacks native ``SupportsStructuredOutput``
     b. ``ObservableLLM`` — event emission
   - ToolRouter ports:
     a. ``ObservableToolRouter`` — event emission

Resulting stack::

    adapter → [user middleware...] → StructuredOutputFallback? → ObservableLLM
"""

from __future__ import annotations

from typing import Any

from hexdag.kernel.logging import get_logger
from hexdag.kernel.ports.llm import SupportsGeneration
from hexdag.kernel.ports.tool_router import ToolRouter

logger = get_logger(__name__)


def _resolve_middleware_class(module_path: str) -> type:
    """Resolve a middleware module path to a class."""
    from hexdag.kernel.resolver import resolve  # lazy: avoid circular import

    return resolve(module_path)


def prepare_ports(
    ports: dict[str, Any],
    middleware_config: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Prepare ports with user middleware + auto-stacked observability.

    Parameters
    ----------
    ports : dict[str, Any]
        Map of port_name → adapter instance.
    middleware_config : dict[str, list[str]] | None
        Optional map of port_name → list of middleware module paths.
        Each middleware class must accept the inner port as its sole
        constructor argument.  Applied inner-to-outer (first item wraps
        the adapter directly).

    Returns
    -------
    dict[str, Any]
        Map of port_name → wrapped port with middleware applied.
    """
    from hexdag.kernel.ports.llm import SupportsStructuredOutput  # lazy: avoid circular import
    from hexdag.stdlib.middleware.observable import ObservableLLM  # lazy: kernel→stdlib boundary
    from hexdag.stdlib.middleware.observable_tool_router import (
        ObservableToolRouter,  # lazy: kernel→stdlib boundary
    )
    from hexdag.stdlib.middleware.structured_output import (
        StructuredOutputFallback,  # lazy: kernel→stdlib boundary
    )

    middleware_config = middleware_config or {}

    prepared: dict[str, Any] = {}
    for name, port in ports.items():
        # Classify port type BEFORE user middleware (which may break isinstance)
        is_llm = isinstance(port, SupportsGeneration)
        is_tool_router = isinstance(port, ToolRouter)
        needs_structured_fallback = is_llm and not isinstance(port, SupportsStructuredOutput)

        # Phase 1: User-declared middleware (from YAML)
        if name in middleware_config:
            for mw_path in middleware_config[name]:
                mw_cls = _resolve_middleware_class(mw_path)
                port = mw_cls(port)
                logger.debug("Applied middleware {} to port '{}'", mw_path, name)

        # Phase 2: Auto-stacked framework middleware (always outermost)
        if is_llm:
            if needs_structured_fallback:
                port = StructuredOutputFallback(port)
            port = ObservableLLM(port)
        elif is_tool_router:
            port = ObservableToolRouter(port)

        prepared[name] = port
    return prepared
