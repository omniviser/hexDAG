"""Visualization utilities for HexDAG.

This is an optional module that requires the 'viz' extra to be installed:
    pip install hexdag[viz]
    or
    uv pip install hexdag[viz]
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hexai.visualization.dag_visualizer import DAGVisualizer


# Check if graphviz is available without importing at module level
def _check_graphviz() -> bool:
    """Check if graphviz is available."""
    try:
        import graphviz  # noqa: F401

        return True
    except ImportError:
        return False


GRAPHVIZ_AVAILABLE = _check_graphviz()


def __getattr__(name: str) -> Any:
    """Lazy import visualization components."""
    if name == "DAGVisualizer":
        try:
            import graphviz  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Graphviz is not installed. Please install with:\n  uv pip install hexdag[viz]"
            ) from e

        from hexai.visualization.dag_visualizer import DAGVisualizer

        return DAGVisualizer

    if name == "render_dag_to_image":
        try:
            import graphviz  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Graphviz is not installed. Please install with:\n  uv pip install hexdag[viz]"
            ) from e

        from hexai.visualization.dag_visualizer import render_dag_to_image

        return render_dag_to_image

    if name == "export_dag_to_dot":
        try:
            import graphviz  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Graphviz is not installed. Please install with:\n  uv pip install hexdag[viz]"
            ) from e

        from hexai.visualization.dag_visualizer import export_dag_to_dot

        return export_dag_to_dot

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["DAGVisualizer", "GRAPHVIZ_AVAILABLE"]
