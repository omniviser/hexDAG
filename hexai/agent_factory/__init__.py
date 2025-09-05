"""Pipeline modules for hexAI framework.

This package contains pre-built pipeline applications that demonstrate the capabilities of the hexAI
framework for specific use cases.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import PipelineCatalog, PipelineDefinition, get_catalog  # noqa: F401
    from .models import (  # noqa: F401
        Ontology,
        OntologyNode,
        OntologyRelation,
        QueryIntent,
        RelationshipType,
        SQLQuery,
    )
    from .yaml_builder import YamlPipelineBuilder  # noqa: F401

_LAZY_MAP: dict[str, tuple[str, str]] = {
    "YamlPipelineBuilder": ("hexai.agent_factory.yaml_builder", "YamlPipelineBuilder"),
    "PipelineCatalog": ("hexai.agent_factory.base", "PipelineCatalog"),
    "PipelineDefinition": ("hexai.agent_factory.base", "PipelineDefinition"),
    "get_catalog": ("hexai.agent_factory.base", "get_catalog"),
    "Ontology": ("hexai.agent_factory.models", "Ontology"),
    "OntologyNode": ("hexai.agent_factory.models", "OntologyNode"),
    "OntologyRelation": ("hexai.agent_factory.models", "OntologyRelation"),
    "QueryIntent": ("hexai.agent_factory.models", "QueryIntent"),
    "RelationshipType": ("hexai.agent_factory.models", "RelationshipType"),
    "SQLQuery": ("hexai.agent_factory.models", "SQLQuery"),
}

__all__ = list(_LAZY_MAP.keys())


def __getattr__(name: str) -> Any:
    if name in _LAZY_MAP:
        module_name, attr = _LAZY_MAP[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr)
        globals()[name] = value  # cache
        return value
    raise AttributeError(f"module {__name__} has no attribute {name}")
