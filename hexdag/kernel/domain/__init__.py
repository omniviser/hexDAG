"""Domain layer exports for the Hex-DAG Agent Framework."""

# DAG primitives - only export what's used in main application code
from hexdag.kernel.domain.dag import DirectedGraph, NodeSpec

__all__ = [
    # DAG primitives (used in main application)
    "DirectedGraph",
    "NodeSpec",
]
