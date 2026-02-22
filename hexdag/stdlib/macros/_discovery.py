"""Auto-discovery of macro aliases for the resolver.

Uses a static registry for consistency with the adapter discovery pattern.

Alias forms
-----------
For a macro class ``ReasoningAgentMacro`` with short name ``reasoning_agent``:

- ``reasoning_agent_macro``   (snake_case of class name)
- ``core:reasoning_agent``    (core-qualified short name)
- ``ReasoningAgentMacro``     (CamelCase, backward compat)

Examples
--------
>>> aliases = discover_macro_aliases()
>>> aliases["core:reasoning_agent"]
'hexdag.stdlib.macros.ReasoningAgentMacro'
>>> aliases["ReasoningAgentMacro"]
'hexdag.stdlib.macros.ReasoningAgentMacro'
"""

from __future__ import annotations

import re
from functools import lru_cache


def _to_snake_case(name: str) -> str:
    """Convert CamelCase to snake_case.

    Examples
    --------
    >>> _to_snake_case("ReasoningAgentMacro")
    'reasoning_agent_macro'
    >>> _to_snake_case("LLMMacro")
    'llm_macro'
    """
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", name)
    return name.lower()


# Static registry: (class_name, full_module_path, short_name_for_core_prefix)
_MACRO_REGISTRY: list[tuple[str, str, str]] = [
    ("ReasoningAgentMacro", "hexdag.stdlib.macros.ReasoningAgentMacro", "reasoning_agent"),
    ("ConversationMacro", "hexdag.stdlib.macros.ConversationMacro", "conversation"),
    ("LLMMacro", "hexdag.stdlib.macros.LLMMacro", "llm_workflow"),
]


@lru_cache(maxsize=1)
def discover_macro_aliases() -> dict[str, str]:
    """Generate all macro alias -> full_module_path mappings.

    Returns
    -------
    dict[str, str]
        Mapping of alias to full module path.

    Examples
    --------
    >>> aliases = discover_macro_aliases()
    >>> "core:reasoning_agent" in aliases
    True
    >>> "ReasoningAgentMacro" in aliases
    True
    >>> "reasoning_agent_macro" in aliases
    True
    """
    aliases: dict[str, str] = {}

    for class_name, full_path, short_name in _MACRO_REGISTRY:
        snake_name = _to_snake_case(class_name)

        # Level 1: snake_case class name (e.g., reasoning_agent_macro)
        aliases[snake_name] = full_path

        # Level 2: core-qualified short name (e.g., core:reasoning_agent)
        aliases[f"core:{short_name}"] = full_path

        # Level 3: CamelCase class name (e.g., ReasoningAgentMacro)
        aliases[class_name] = full_path

    return aliases


def get_known_macro_aliases() -> frozenset[str]:
    """Get all valid macro alias names.

    Returns
    -------
    frozenset[str]
        Set of all valid macro aliases.
    """
    return frozenset(discover_macro_aliases().keys())
