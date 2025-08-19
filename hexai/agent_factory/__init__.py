"""
Agent Factory subpackage (lazy, conditional imports).

This package contains optional pipeline utilities and domain models.
Attributes are imported on first access to avoid importing optional dependencies eagerly.

Examples:
    from hexai.agent_factory import YamlPipelineBuilder
    from hexai.agent_factory import PipelineCatalog, PipelineDefinition, get_catalog
    from hexai.agent_factory import (
        Ontology, OntologyNode, OntologyRelation, QueryIntent, RelationshipType, SQLQuery
    )
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_MAP: dict[str, tuple[str, str]] = {
    # Builders
    "YamlPipelineBuilder": ("hexai.agent_factory.yaml_builder", "YamlPipelineBuilder"),
    # Catalog/base
    "PipelineCatalog": ("hexai.agent_factory.base", "PipelineCatalog"),
    "PipelineDefinition": ("hexai.agent_factory.base", "PipelineDefinition"),
    "get_catalog": ("hexai.agent_factory.base", "get_catalog"),
    # Models
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
        globals()[name] = value  # cache for subsequent access
        return value
    raise AttributeError(f"module {__name__} has no attribute {name}")
