"""Base mixin for auto-registering adapters via ``__init_subclass__``."""

from __future__ import annotations

from typing import Any, ClassVar


class HexDAGAdapter:
    """Mixin that auto-registers adapters when ``yaml_alias`` is provided.

    Examples
    --------
    >>> class MyAdapter(HexDAGAdapter, yaml_alias="my_adapter", port="llm"):
    ...     pass
    >>> "my_adapter" in HexDAGAdapter._registry
    True
    >>> "llm:my_adapter" in HexDAGAdapter._registry
    True
    """

    _registry: ClassVar[dict[str, str]] = {}
    """Auto-populated by ``__init_subclass__``: alias -> full module path."""

    def __init_subclass__(
        cls,
        *,
        yaml_alias: str | None = None,
        port: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Register subclass in the alias registry."""
        super().__init_subclass__(**kwargs)
        if yaml_alias:
            full_path = f"{cls.__module__}.{cls.__qualname__}"
            HexDAGAdapter._registry[yaml_alias] = full_path
            # Also register CamelCase class name
            HexDAGAdapter._registry[cls.__name__] = full_path
            # Port-qualified alias: llm:openai
            if port:
                HexDAGAdapter._registry[f"{port}:{yaml_alias}"] = full_path

            # Push aliases into the kernel registry immediately
            from hexdag.kernel._alias_registry import (
                register_builtin_aliases,  # lazy: avoid import at class-definition time
            )

            aliases: dict[str, str] = {yaml_alias: full_path, cls.__name__: full_path}
            if port:
                aliases[f"{port}:{yaml_alias}"] = full_path
            register_builtin_aliases(aliases)
