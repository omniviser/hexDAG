"""Pipeline modules for hexAI framework.

This package contains pre-built pipeline applications that demonstrate the capabilities of the hexAI
framework for specific use cases.

Note: YamlPipelineBuilder requires the 'cli' extra to be installed:
    pip install hexdag[cli]
    or
    uv pip install hexdag[cli]
"""

from typing import TYPE_CHECKING, Any

# Always available imports (no external dependencies)
from .base import PipelineCatalog, PipelineDefinition, get_catalog
from .models import (
    Ontology,
    OntologyNode,
    OntologyRelation,
    QueryIntent,
    RelationshipType,
    SQLQuery,
)

if TYPE_CHECKING:
    from .yaml_builder import YamlPipelineBuilder


def _check_yaml() -> bool:
    """Check if PyYAML is available."""
    try:
        import yaml

        del yaml  # Remove from namespace after checking
        return True
    except ImportError:
        return False


YAML_AVAILABLE = _check_yaml()


def __getattr__(name: str) -> Any:
    """Lazy import for YAML-dependent components."""
    if name == "YamlPipelineBuilder":
        try:
            import yaml

            del yaml  # Verify it's available but don't keep in namespace
        except ImportError as e:
            raise ImportError(
                "PyYAML is not installed. Please install with:\n"
                "  pip install hexdag[cli]\n"
                "  or\n"
                "  uv pip install hexdag[cli]"
            ) from e

        from .yaml_builder import YamlPipelineBuilder

        return YamlPipelineBuilder

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Agent Factory System
    "PipelineDefinition",
    "PipelineCatalog",
    "get_catalog",
    "YamlPipelineBuilder",
    # Agent Factory Models
    "Ontology",
    "OntologyNode",
    "OntologyRelation",
    # Query Models
    "QueryIntent",
    "SQLQuery",
    "RelationshipType",
    # Availability flags
    "YAML_AVAILABLE",
]
