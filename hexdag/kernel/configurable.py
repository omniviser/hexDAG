"""Configurable interface and base class for plugins and adapters.

This module provides:
1. ConfigurableComponent - Protocol for components with configuration
2. ConfigurableAdapter - Base class that implements the protocol and eliminates boilerplate
3. ConfigurableNode - Base class for node factories with configuration
4. ConfigurablePolicy - Base class for policies with configuration
5. ConfigurableMacro - Base class for macros with configuration
6. SecretField - Helper for declaring secret configuration fields
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field  # noqa: TC002

from hexdag.kernel.logging import get_logger

logger = get_logger(__name__)


def _extract_config_fields(kwargs: dict[str, Any], config_class: type[BaseModel]) -> dict[str, Any]:
    """Extract config fields from kwargs that match the config class schema.

    Parameters
    ----------
    kwargs : dict[str, Any]
        Keyword arguments containing config values
    config_class : type[BaseModel]
        Pydantic model class defining the config schema

    Returns
    -------
    dict[str, Any]
        Dictionary containing only the fields defined in config_class
    """
    return {
        field_name: kwargs[field_name]
        for field_name in config_class.model_fields
        if field_name in kwargs
    }


# Legacy Config base classes removed - use simplified decorator pattern instead
# See CLAUDE.md and SIMPLIFIED_PATTERN.md for migration guide


class MacroConfig(BaseModel):
    """Base configuration class for all macros.

    Macros should define a nested Config class inheriting from this.
    This enables:
    - Type-safe configuration
    - YAML schema generation
    - Runtime validation
    - Expansion strategy configuration

    Examples
    --------
    Define a macro configuration class::

        class ResearchMacroConfig(MacroConfig):
            depth: int = 3
            enable_synthesis: bool = True

        config = ResearchMacroConfig(depth=5)
        assert config.depth == 5
    """

    model_config = ConfigDict(frozen=True)


def SecretField(
    env_var: str,
    memory_key: str | None = None,
    default: Any = None,
    description: str | None = None,
    **field_kwargs: Any,
) -> Any:
    """Create a secret field with auto-resolution from Memory or environment.

    This helper marks a Pydantic field as a secret, which enables:
    1. Auto-hiding in logs/repr (uses Pydantic SecretStr)
    2. Auto-resolution from Memory (secret:KEY) via orchestrator
    3. Fallback to environment variable if not in Memory
    4. Clear documentation of secret requirements

    Parameters
    ----------
    env_var : str
        Environment variable name to read from (e.g., "OPENAI_API_KEY")
    memory_key : str | None, optional
        Key to use in Memory port (defaults to env_var).
        Will be prefixed with "secret:" automatically.
    default : Any, optional
        Default value if not found (default: None)
    description : str | None, optional
        Field description for schema documentation
    **field_kwargs : Any
        Additional Pydantic Field() parameters

    Returns
    -------
    Any
        Pydantic Field with secret metadata (typed as Any for use in assignments)

    Examples
    --------
    >>> from pydantic import BaseModel
    >>> class MyAdapterConfig(BaseModel):
    ...     api_key: SecretStr | None = SecretField(
    ...         env_var="OPENAI_API_KEY",
    ...         description="OpenAI API key"
    ...     )
    ...     timeout: float = 30.0
    """
    return Field(
        default=default,
        description=description,
        json_schema_extra={
            "secret": True,
            "env_var": env_var,
            "memory_key": memory_key or env_var,
        },
        **field_kwargs,
    )


@runtime_checkable
class ConfigurableComponent(Protocol):
    """Protocol for components that support configuration.

    Any adapter or plugin that implements this protocol will be
    automatically discoverable by the CLI config generation system.
    """

    @classmethod
    def get_config_class(cls) -> type[BaseModel]:
        """Return the Pydantic model class that defines configuration schema.

        Returns
        -------
        type[BaseModel]
            A Pydantic model class with field definitions, defaults, and validation
        """
        ...


class ConfigurableMacro:
    """Base class for macros with configuration support.

    Macros are pipeline templates that expand into DirectedGraph subgraphs.
    They are first-class registry components like nodes, adapters, and policies.

    Key Concepts
    ------------
    - Macros live at a higher abstraction level than nodes
    - Nodes are atomic operations (single LLM call, single function)
    - Macros are compositions (multi-step workflows)
    - Macros expand to graphs of nodes at build time or runtime

    Architecture
    ------------
    ConfigurableMacro provides:
    1. Type-safe configuration via MacroConfig subclasses
    2. Consistent expansion interface via expand() method
    3. Registry integration via @macro decorator
    4. Support for both static and dynamic expansion strategies

    Subclasses must:
    - Define a nested Config class inheriting from MacroConfig
    - Implement expand() method that returns DirectedGraph

    Examples
    --------
    >>> from hexdag.kernel.configurable import ConfigurableMacro, MacroConfig
    >>> from hexdag.kernel.domain.dag import DirectedGraph
    >>>
    >>> class ResearchMacroConfig(MacroConfig):
    ...     depth: int = 3
    ...     enable_synthesis: bool = True
    >>>
    >>> class ResearchMacro(ConfigurableMacro):
    ...     Config = ResearchMacroConfig
    ...
    ...     def expand(self, instance_name, inputs, dependencies):
    ...         # Build and return DirectedGraph
    ...         return DirectedGraph([...])
    """

    Config: type[MacroConfig]

    def __init__(self, **kwargs: Any) -> None:
        """Initialize macro with configuration.

        Parameters
        ----------
        **kwargs : Any
            Configuration options matching Config schema fields
        """
        if not hasattr(self.__class__, "Config"):
            raise AttributeError(
                f"{self.__class__.__name__} must define a nested Config class (MacroConfig)"
            )

        config_data = _extract_config_fields(kwargs, self.Config)

        self.config = self.Config(**config_data)
        self._extra_kwargs = {k: v for k, v in kwargs.items() if k not in self.Config.model_fields}

    def expand(
        self,
        instance_name: str,
        inputs: dict[str, Any],
        dependencies: list[str],
    ) -> Any:  # Returns DirectedGraph but avoid circular import
        """Expand macro into a concrete subgraph.

        This is the core method that subclasses must implement.
        It transforms the macro template into an actual DirectedGraph
        with concrete nodes.

        Parameters
        ----------
        instance_name : str
            Unique name for this macro instance (used as prefix for generated nodes).
            Example: "deep_research" → generates "deep_research_step_1", etc.
        inputs : dict[str, Any]
            Input values to bind to macro parameters.
            Example: {"topic": "AI safety", "depth": 5}
        dependencies : list[str]
            External node names that this macro instance depends on.
            The macro's entry nodes will be connected to these.
            Example: ["query_parser", "validator"]

        Returns
        -------
        DirectedGraph
            Subgraph containing the expanded nodes with proper dependencies

        Raises
        ------
        NotImplementedError
            If subclass doesn't implement this method
        ValueError
            If inputs don't match macro parameter requirements
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement expand() method")

    def validate_inputs(
        self, inputs: dict[str, Any], required: list[str], optional: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate and normalize macro inputs.

        Helper method for subclasses to validate inputs against requirements.

        Parameters
        ----------
        inputs : dict[str, Any]
            Provided input values
        required : list[str]
            Names of required input parameters
        optional : dict[str, Any]
            Optional parameters with their default values

        Returns
        -------
        dict[str, Any]
            Validated and normalized inputs (with defaults applied)

        Raises
        ------
        ValueError
            If required inputs are missing
        """
        # Check required inputs
        if missing := [name for name in required if name not in inputs]:
            raise ValueError(
                f"Missing required inputs for {self.__class__.__name__}: {', '.join(missing)}"
            )

        # Apply defaults for optional inputs
        return {**optional, **inputs}

    @classmethod
    def get_config_class(cls) -> type[BaseModel]:
        """Get the configuration model class."""
        return cls.Config

    def __repr__(self) -> str:
        """Readable representation for debugging."""
        config_dict = self.config.model_dump() if hasattr(self.config, "model_dump") else {}
        return f"{self.__class__.__name__}(config={config_dict})"
