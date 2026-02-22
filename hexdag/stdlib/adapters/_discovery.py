"""Auto-discovery of adapter aliases for the resolver.

Uses a static registry to avoid importing heavy adapter dependencies
(openai, anthropic, asyncpg) at bootstrap time. Heavy deps are only
imported when ``resolve()`` actually loads the class.

Alias forms
-----------
For an adapter class ``MockLLM`` with port type ``llm`` and short name ``mock``:

- ``mock_llm``       (snake_case of class name)
- ``llm:mock``       (port-qualified)
- ``MockLLM``        (CamelCase, backward compat)

Examples
--------
>>> aliases = discover_adapter_aliases()
>>> aliases["mock_llm"]
'hexdag.stdlib.adapters.mock.MockLLM'
>>> aliases["llm:mock"]
'hexdag.stdlib.adapters.mock.MockLLM'
>>> aliases["MockLLM"]
'hexdag.stdlib.adapters.mock.MockLLM'
"""

from __future__ import annotations

import re
from functools import lru_cache


def _to_snake_case(name: str) -> str:
    """Convert CamelCase to snake_case.

    Examples
    --------
    >>> _to_snake_case("MockLLM")
    'mock_llm'
    >>> _to_snake_case("OpenAIAdapter")
    'open_ai_adapter'
    >>> _to_snake_case("InMemoryMemory")
    'in_memory_memory'
    """
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", name)
    return name.lower()


# Static registry: (class_name, full_module_path, port_type, short_name)
# short_name is used in port-qualified aliases (e.g., llm:mock)
_ADAPTER_REGISTRY: list[tuple[str, str, str, str]] = [
    # Mock adapters
    ("MockLLM", "hexdag.stdlib.adapters.mock.MockLLM", "llm", "mock"),
    ("MockDatabaseAdapter", "hexdag.stdlib.adapters.mock.MockDatabaseAdapter", "database", "mock"),
    ("MockEmbedding", "hexdag.stdlib.adapters.mock.MockEmbedding", "embedding", "mock"),
    # OpenAI
    ("OpenAIAdapter", "hexdag.stdlib.adapters.openai.OpenAIAdapter", "llm", "openai"),
    # Anthropic
    ("AnthropicAdapter", "hexdag.stdlib.adapters.anthropic.AnthropicAdapter", "llm", "anthropic"),
    # Memory adapters
    ("InMemoryMemory", "hexdag.stdlib.adapters.memory.InMemoryMemory", "memory", "in_memory"),
    ("FileMemoryAdapter", "hexdag.stdlib.adapters.memory.FileMemoryAdapter", "memory", "file"),
    (
        "SQLiteMemoryAdapter",
        "hexdag.stdlib.adapters.memory.SQLiteMemoryAdapter",
        "memory",
        "sqlite",
    ),
    # Database adapters
    ("SQLiteAdapter", "hexdag.stdlib.adapters.database.sqlite.SQLiteAdapter", "database", "sqlite"),  # noqa: E501
    (
        "PgVectorAdapter",
        "hexdag.stdlib.adapters.database.pgvector.PgVectorAdapter",
        "database",
        "pgvector",
    ),
    (
        "SQLAlchemyAdapter",
        "hexdag.stdlib.adapters.database.sqlalchemy.SQLAlchemyAdapter",
        "database",
        "sqlalchemy",
    ),
    ("CsvAdapter", "hexdag.stdlib.adapters.database.csv.CsvAdapter", "database", "csv"),
    # Secret adapters
    ("LocalSecretAdapter", "hexdag.stdlib.adapters.secret.LocalSecretAdapter", "secret", "local"),
    # ToolRouter lives in kernel/ports but is treated as an adapter
    ("ToolRouter", "hexdag.kernel.ports.tool_router.ToolRouter", "tool_router", "default"),
]

# Backward-compatibility aliases
_COMPAT_ALIASES: dict[str, str] = {
    "UnifiedToolRouter": "hexdag.kernel.ports.tool_router.ToolRouter",
    "FunctionToolRouter": "hexdag.kernel.ports.tool_router.ToolRouter",
    "FunctionBasedToolRouter": "hexdag.kernel.ports.tool_router.ToolRouter",
}


@lru_cache(maxsize=1)
def discover_adapter_aliases() -> dict[str, str]:
    """Generate all adapter alias -> full_module_path mappings.

    Returns
    -------
    dict[str, str]
        Mapping of alias to full module path.

    Examples
    --------
    >>> aliases = discover_adapter_aliases()
    >>> "mock_llm" in aliases
    True
    >>> "llm:mock" in aliases
    True
    >>> "MockLLM" in aliases
    True
    """
    aliases: dict[str, str] = {}

    for class_name, full_path, port_type, short_name in _ADAPTER_REGISTRY:
        snake_name = _to_snake_case(class_name)

        # Level 1: snake_case class name (e.g., mock_llm)
        aliases[snake_name] = full_path

        # Level 2: port-qualified (e.g., llm:mock)
        aliases[f"{port_type}:{short_name}"] = full_path

        # Level 3: CamelCase class name (e.g., MockLLM)
        aliases[class_name] = full_path

    # Backward-compat aliases
    aliases.update(_COMPAT_ALIASES)

    return aliases


def get_known_adapter_aliases() -> frozenset[str]:
    """Get all valid adapter alias names.

    Returns
    -------
    frozenset[str]
        Set of all valid adapter aliases.
    """
    return frozenset(discover_adapter_aliases().keys())
