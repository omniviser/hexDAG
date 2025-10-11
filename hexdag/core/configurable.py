"""Configurable interface and base class for plugins and adapters.

This module provides:
1. ConfigurableComponent - Protocol for components with configuration
2. ConfigurableAdapter - Base class that implements the protocol and eliminates boilerplate
3. ConfigurableNode - Base class for node factories with configuration
4. ConfigurablePolicy - Base class for policies with configuration
5. SecretField - Helper for declaring secret configuration fields
"""

from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, SecretStr  # noqa: TC002

from hexdag.core.logging import get_logger

logger = get_logger(__name__)


class AdapterConfig(BaseModel):
    """Base configuration class for all adapters.

    This base class provides common configuration behavior:
    - Frozen models (immutable after creation) for safety
    - Consistent configuration across all adapters

    Subclasses should define their specific configuration fields.

    Examples
    --------
    >>> class MyAdapterConfig(AdapterConfig):
    ...     api_key: str
    ...     timeout: float = 30.0
    >>> config = MyAdapterConfig(api_key="test")
    >>> config.timeout
    30.0
    >>> config.timeout = 60  # doctest: +SKIP
    ValidationError: Instance is frozen
    """

    model_config = ConfigDict(frozen=True)


class NodeConfig(BaseModel):
    """Base configuration class for all node factories.

    Node factories should define a nested Config class inheriting from this.
    This enables:
    - Type-safe configuration
    - YAML schema generation
    - Runtime validation

    Examples
    --------
    >>> class LLMNodeConfig(NodeConfig):
    ...     template: str
    ...     max_tokens: int = 1000
    >>> config = LLMNodeConfig(template="Analyze: {{input}}")
    >>> config.max_tokens
    1000
    """

    model_config = ConfigDict(frozen=True)


class PolicyConfig(BaseModel):
    """Base configuration class for all policies.

    Policies should define a nested Config class inheriting from this.

    Examples
    --------
    >>> class RetryPolicyConfig(PolicyConfig):
    ...     max_retries: int = 3
    ...     backoff: str = "exponential"
    >>> config = RetryPolicyConfig(max_retries=5)
    >>> config.backoff
    'exponential'
    """

    model_config = ConfigDict(frozen=True)


class ExecutorConfig(BaseModel):
    """Base configuration class for all executors.

    Executors should define a nested Config class inheriting from this.
    This enables:
    - Type-safe configuration
    - YAML schema generation
    - Runtime validation

    Examples
    --------
    >>> class LocalExecutorConfig(ExecutorConfig):
    ...     max_concurrent_nodes: int = 10
    ...     strict_validation: bool = False
    >>> config = LocalExecutorConfig(max_concurrent_nodes=5)
    >>> config.max_concurrent_nodes
    5
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
    >>> class MyAdapterConfig(AdapterConfig):
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


class ConfigurableAdapter:
    """Base class for adapters with automatic config management.

    This class implements the ConfigurableComponent protocol and eliminates
    boilerplate by automating config extraction, validation, and management.

    Eliminates boilerplate by:
    1. Automatically extracting config from **kwargs
    2. Creating and validating Pydantic config
    3. Making config available via self.config (explicit access)

    Subclasses must:
    - Define a nested Config class (Pydantic model)
    - Explicitly access config fields via self.config.field_name

    Examples
    --------
    Before (boilerplate)::

        class MyAdapter:
            class Config(BaseModel):
                api_key: str
                timeout: float = 30.0

            def __init__(self, **kwargs):
                # 15 lines of boilerplate
                config_data = {}
                for field_name in self.Config.model_fields:
                    if field_name in kwargs:
                        config_data[field_name] = kwargs[field_name]
                config = self.Config(**config_data)
                self.config = config

    After (clean + explicit)::

        class MyAdapter(ConfigurableAdapter):
            class Config(BaseModel):
                api_key: str
                timeout: float = 30.0

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                # Now use self.config.api_key (clear where it comes from!)

    Attributes
    ----------
    config : BaseModel
        Validated configuration instance. Access fields via self.config.field_name
        Type checkers will see this as the specific Config type in subclasses.
    """

    # Subclass must override this with their specific Config class
    Config: type[BaseModel]

    def __init__(self, **kwargs: Any) -> None:
        """Initialize adapter with automatic config and secret management.

        Parameters
        ----------
        **kwargs : Any
            Configuration options matching Config schema fields.
            Extra kwargs (not in Config) are stored for get_extra_kwarg().
            Special kwargs:
            - memory: Memory port for runtime secret resolution (optional)

        Raises
        ------
        AttributeError
            If Config class is not defined
        """
        # Validate Config class exists
        if not hasattr(self.__class__, "Config"):
            raise AttributeError(
                f"{self.__class__.__name__} must define a nested Config class (Pydantic model)"
            )

        # Extract memory for secret resolution (don't pass to config)
        self._memory = kwargs.pop("memory", None)

        # Auto-resolve secrets from environment (synchronous)
        kwargs = self._resolve_secrets_from_env(kwargs)

        # Extract config fields from kwargs
        config_data = {
            field_name: kwargs[field_name]
            for field_name in self.Config.model_fields
            if field_name in kwargs
        }

        # Create and validate config (accessible via self.config.field_name)
        self.config = self.Config(**config_data)

        # Store extra kwargs not in config schema (for API-specific params)
        self._extra_kwargs = {k: v for k, v in kwargs.items() if k not in self.Config.model_fields}

    def _resolve_secrets_from_env(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Resolve secret fields from environment variables.

        Resolution order for secret fields:
        1. Explicit kwarg (highest priority) - kwargs['api_key'] = "sk-..."
        2. Environment variable - os.getenv("OPENAI_API_KEY")
        3. Field default - None (will be set to default)

        Parameters
        ----------
        kwargs : dict[str, Any]
            Input keyword arguments

        Returns
        -------
        dict[str, Any]
            Updated kwargs with resolved secrets
        """
        for field_name, field_info in self.Config.model_fields.items():
            # Check if field is marked as secret
            extras = field_info.json_schema_extra
            if not isinstance(extras, dict):
                continue
            if not extras.get("secret"):
                continue

            # Skip if already provided explicitly
            if field_name in kwargs and kwargs[field_name]:
                # Wrap in SecretStr if it's a plain string
                if isinstance(kwargs[field_name], str):
                    kwargs[field_name] = SecretStr(kwargs[field_name])
                continue

            # Try environment variable
            env_var = extras.get("env_var")
            if env_var and isinstance(env_var, str):
                env_value = os.getenv(env_var)
                if env_value:
                    kwargs[field_name] = SecretStr(env_value)
                    logger.debug(
                        "Resolved secret field '{field}' from env var {env}",
                        field=field_name,
                        env=env_var,
                    )

        return kwargs

    @classmethod
    def get_config_class(cls) -> type[BaseModel]:
        """Get the configuration model class.

        Returns
        -------
        type[BaseModel]
            The Config class defined in this adapter
        """
        return cls.Config

    def get_extra_kwarg(self, key: str, default: Any = None) -> Any:
        """Get extra kwarg not in config schema.

        Use this for adapter-specific parameters not defined in Config
        (e.g., OpenAI's 'organization', 'base_url').

        Parameters
        ----------
        key : str
            The key to retrieve
        default : Any
            Default value if key not found

        Returns
        -------
        Any
            The value or default

        Examples
        --------
        Example usage::

            # In OpenAI adapter
            if org := self.get_extra_kwarg("organization"):
                client_kwargs["organization"] = org
        """
        return self._extra_kwargs.get(key, default)

    def get_config_dict(self) -> dict[str, Any]:
        """Export config as dictionary for debugging/serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of the configuration

        Examples
        --------
        >>> adapter.get_config_dict()  # doctest: +SKIP
        {'api_key': 'sk-...', 'timeout': 30.0}
        """
        return self.config.model_dump()

    def __repr__(self) -> str:
        """Readable representation showing adapter class and config.

        Returns
        -------
        str
            String representation like: MyAdapter(api_key='***', timeout=30.0)

        Examples
        --------
        >>> adapter = MyAdapter(api_key='secret', timeout=30.0)  # doctest: +SKIP
        >>> repr(adapter)  # doctest: +SKIP
        "MyAdapter(api_key='***', timeout=30.0)"
        """
        config_items = []
        for key, value in self.config.model_dump().items():
            # Mask sensitive fields (api_key, password, secret, token)
            if any(
                sensitive in key.lower() for sensitive in ["key", "password", "secret", "token"]
            ):
                display_value = "'***'"
            else:
                display_value = repr(value)
            config_items.append(f"{key}={display_value}")

        config_str = ", ".join(config_items)
        return f"{self.__class__.__name__}({config_str})"


class ConfigurableNode:
    """Base class for node factories with optional configuration support.

    Similar to ConfigurableAdapter but for node factories. Enables:
    - Type-safe configuration via nested Config class (optional)
    - YAML schema generation
    - Runtime validation
    - Cooperative multiple inheritance via super().__init__()

    Subclasses should use cooperative multiple inheritance by calling super().__init__(**kwargs)
    which will properly initialize all parent classes in MRO order.

    Config class is OPTIONAL - only define it if you have actual configuration fields.
    Don't create empty Config classes for "future extensibility" (YAGNI principle).

    Subclasses must:
    - Implement __call__() method that returns NodeSpec
    - Call super().__init__(**kwargs) if they define custom __init__
    - Optionally define a Config class if they have configuration fields

    Examples
    --------
    >>> from hexdag.core.configurable import ConfigurableNode, NodeConfig
    >>> # Node WITH configuration
    >>> class MyNodeConfig(NodeConfig):
    ...     template: str
    ...     max_tokens: int = 1000
    >>> class ConfiguredNode(ConfigurableNode):  # doctest: +SKIP
    ...     Config = MyNodeConfig
    ...     def __call__(self, name: str, **kwargs):
    ...         # Access self.config.template, self.config.max_tokens
    ...         pass
    >>> # Node WITHOUT configuration (dynamic nodes)
    >>> class DynamicNode(ConfigurableNode):  # doctest: +SKIP
    ...     # No Config class needed!
    ...     def __call__(self, name: str, **kwargs):
    ...         # All config passed via kwargs to __call__
    ...         pass
    """

    Config: type[NodeConfig] | None = None

    def __init__(self, **kwargs: Any) -> None:
        """Initialize node factory with optional configuration.

        Parameters
        ----------
        **kwargs : Any
            Configuration options matching Config schema fields (if Config is defined)
        """
        # Check if Config class is defined
        if not hasattr(self.__class__, "Config") or self.__class__.Config is None:
            # No Config class - node is dynamic, all config via __call__ kwargs
            self.config = None
            self._extra_kwargs = kwargs
            return

        # Type narrowing: Config is not None at this point
        config_class = self.__class__.Config

        # Extract config fields
        config_data = {
            field_name: kwargs[field_name]
            for field_name in config_class.model_fields
            if field_name in kwargs
        }

        self.config = config_class(**config_data)
        self._extra_kwargs = {k: v for k, v in kwargs.items() if k not in config_class.model_fields}

    @classmethod
    def get_config_class(cls) -> type[BaseModel] | None:
        """Get the configuration model class, or None if no config."""
        return cls.Config if hasattr(cls, "Config") else None


class ConfigurablePolicy:
    """Base class for policies with configuration support.

    Similar to ConfigurableAdapter but for policies. Enables:
    - Type-safe configuration via nested Config class
    - YAML schema generation
    - Runtime validation

    Subclasses must:
    - Define a nested Config class inheriting from PolicyConfig
    - Implement evaluate() method

    Examples
    --------
    >>> from hexdag.core.configurable import ConfigurablePolicy, PolicyConfig
    >>> class RetryConfig(PolicyConfig):
    ...     max_retries: int = 3
    ...     backoff: str = "exponential"
    >>> class RetryPolicy(ConfigurablePolicy):
    ...     Config = RetryConfig
    ...     async def evaluate(self, context):
    ...         # Access self.config.max_retries, self.config.backoff
    ...         pass
    """

    Config: type[PolicyConfig]

    def __init__(self, **kwargs: Any) -> None:
        """Initialize policy with configuration.

        Parameters
        ----------
        **kwargs : Any
            Configuration options matching Config schema fields
        """
        if not hasattr(self.__class__, "Config"):
            raise AttributeError(
                f"{self.__class__.__name__} must define a nested Config class (PolicyConfig)"
            )

        # Extract config fields
        config_data = {
            field_name: kwargs[field_name]
            for field_name in self.Config.model_fields
            if field_name in kwargs
        }

        self.config = self.Config(**config_data)
        self._extra_kwargs = {k: v for k, v in kwargs.items() if k not in self.Config.model_fields}

    @classmethod
    def get_config_class(cls) -> type[BaseModel]:
        """Get the configuration model class."""
        return cls.Config
