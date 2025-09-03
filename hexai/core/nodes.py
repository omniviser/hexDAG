"""Core node components for hexDAG framework.

These nodes are automatically registered when the registry initializes.
They are always available as part of the core framework.
"""

from __future__ import annotations

import logging
from typing import Any

from hexai.core.registry import node

logger = logging.getLogger(__name__)


@node(namespace="core", replaceable=False)
class PassThroughNode:
    """Passes data through without modification.

    Useful for testing and as a placeholder in DAGs.
    """

    def execute(self, data: Any) -> Any:
        """Pass data through unchanged."""
        return data


@node(namespace="core", replaceable=False)
class LoggingNode:
    """Logs data passing through the node.

    Useful for debugging and monitoring DAG execution.
    """

    def __init__(self, log_level: str = "INFO"):
        """Initialize logging node.

        Parameters
        ----------
        log_level : str
            Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)

    def execute(self, data: Any) -> Any:
        """Log data and pass it through."""
        logger.log(self.log_level, f"Data passing through: {data}")
        return data


# Note: These components are automatically registered when this module
# is imported by the registry during initialization. No explicit
# registration code is needed here.
