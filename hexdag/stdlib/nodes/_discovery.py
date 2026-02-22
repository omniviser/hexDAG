"""Auto-discovery of node factories in builtin.nodes package.

This module provides automatic discovery of all BaseNodeFactory subclasses,
eliminating the need to manually register node types in multiple places.

Usage
-----
Adding a new node only requires creating the node file:

    # hexdag/stdlib/nodes/my_node.py
    class MyNode(BaseNodeFactory):
        def __call__(self, name: str, **kwargs) -> NodeSpec:
            ...

The node is automatically available in YAML as:
- my_node
- core:my_node
- core:my
"""

from __future__ import annotations

import importlib
import pkgutil
import re
from functools import lru_cache


def _to_snake_case(name: str) -> str:
    """Convert CamelCase to snake_case.

    Examples
    --------
    >>> _to_snake_case("LLMNode")
    'llm_node'
    >>> _to_snake_case("ReActAgentNode")
    're_act_agent_node'
    >>> _to_snake_case("DataNode")
    'data_node'
    """
    # Handle acronyms like LLM, ReAct, then standard CamelCase
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", name)
    return name.lower()


@lru_cache(maxsize=1)
def discover_node_factories() -> dict[str, str]:
    """Discover all BaseNodeFactory subclasses and generate aliases.

    Returns mapping: {alias: full_module_path}

    Examples
    --------
    >>> aliases = discover_node_factories()
    >>> "llm_node" in aliases
    True
    >>> "core:llm_node" in aliases
    True
    >>> "core:llm" in aliases
    True
    """
    # Import here to avoid circular imports
    from hexdag.stdlib.nodes.base_node_factory import BaseNodeFactory

    aliases: dict[str, str] = {}
    package = importlib.import_module("hexdag.stdlib.nodes")

    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.name.startswith("_"):
            continue

        try:
            module = importlib.import_module(f"hexdag.stdlib.nodes.{module_info.name}")
        except ImportError:
            continue

        for attr_name in dir(module):
            if attr_name.startswith("_"):
                continue
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseNodeFactory)
                and attr is not BaseNodeFactory
                and attr.__name__ == attr_name  # skip module-level aliases
            ):
                full_path = f"hexdag.stdlib.nodes.{attr_name}"
                snake_name = _to_snake_case(attr_name)

                # Generate all alias forms
                aliases[snake_name] = full_path  # llm_node
                aliases[f"core:{snake_name}"] = full_path  # core:llm_node

                # Also add without _node suffix for convenience
                if snake_name.endswith("_node"):
                    base = snake_name[:-5]
                    aliases[f"core:{base}"] = full_path  # core:llm

                # Register __aliases__ declared on the class itself
                for alias in getattr(attr, "__aliases__", ()):
                    aliases[alias] = full_path
                    aliases[f"core:{alias}"] = full_path
                    if alias.endswith("_node"):
                        aliases[f"core:{alias[:-5]}"] = full_path

    return aliases


def get_known_node_types() -> frozenset[str]:
    """Get all valid node type names for YAML validation.

    Returns
    -------
    frozenset[str]
        Set of all valid node type aliases
    """
    return frozenset(discover_node_factories().keys())
