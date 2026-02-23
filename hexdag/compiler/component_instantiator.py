"""Component instantiator — includes Phase 3b of the rendering pipeline.

Handles deferred ``${VAR}`` resolution: secret-like variables preserved
during Phase 1 are resolved here at adapter/policy instantiation time.

Error messages for deferred resolution are prefixed with
``[Phase 3b: Deferred Secret Resolution]``.

Also handles: parsing YAML component specs, resolving module paths,
instantiating adapters and policies with configuration.
"""

import os
import re
from collections import namedtuple
from typing import Any

from hexdag.kernel.exceptions import ComponentInstantiationError  # noqa: F401
from hexdag.kernel.logging import get_logger
from hexdag.kernel.resolver import resolve

logger = get_logger(__name__)

# Pattern for deferred environment variables: ${VAR} or ${VAR:default}
_DEFERRED_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::([^}]*))?\}")

# Simple namedtuple for component specification
ComponentSpec = namedtuple("ComponentSpec", ["module_path", "params"])


def _resolve_deferred_env_vars(params: dict[str, Any]) -> dict[str, Any]:
    """Resolve deferred ${VAR} syntax in parameters at runtime.

    This completes the deferred secret resolution workflow where secrets like
    ${OPENAI_API_KEY} are preserved at YAML build-time and resolved here
    at adapter instantiation time.

    Parameters
    ----------
    params : dict[str, Any]
        Parameters potentially containing ${VAR} or ${VAR:default} syntax

    Returns
    -------
    dict[str, Any]
        Parameters with environment variables resolved

    Note
    ----
    Raises ComponentInstantiationError (via _resolve_string_value) if a
    required environment variable is not set and has no default.

    Examples
    --------
    >>> import os
    >>> os.environ["TEST_KEY"] = "secret123"
    >>> _resolve_deferred_env_vars({"api_key": "${TEST_KEY}"})
    {'api_key': 'secret123'}
    >>> _resolve_deferred_env_vars({"model": "${MODEL:gpt-4}"})
    {'model': 'gpt-4'}
    """
    resolved: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, str):
            resolved[key] = _resolve_string_value(value)
        elif isinstance(value, dict):
            resolved[key] = _resolve_deferred_env_vars(value)
        elif isinstance(value, list):
            resolved[key] = [
                _resolve_string_value(v)
                if isinstance(v, str)
                else _resolve_deferred_env_vars(v)
                if isinstance(v, dict)
                else v
                for v in value
            ]
        else:
            resolved[key] = value
    return resolved


def _resolve_string_value(value: str) -> str:
    """Resolve ${VAR} or ${VAR:default} patterns in a string value.

    Parameters
    ----------
    value : str
        String potentially containing ${VAR} patterns

    Returns
    -------
    str
        String with all ${VAR} patterns resolved

    Note
    ----
    Raises ComponentInstantiationError (via nested replacer) if a required
    environment variable is not set and has no default.
    """

    def replacer(match: re.Match[str]) -> str:
        var_name = match.group(1)
        default = match.group(2)
        env_value = os.environ.get(var_name)

        if env_value is None:
            if default is not None:
                logger.debug(f"Using default value for ${{{var_name}}}")
                return default
            raise ComponentInstantiationError(
                f"[Phase 3b: Deferred Secret Resolution] "
                f"Environment variable '{var_name}' is not set and no default provided.\n"
                f"  In: {value}\n"
                f"  Hint: This variable was deferred from build-time (secret pattern match). "
                f"Set the environment variable before running the pipeline, "
                f"or use ${{{var_name}:default}} syntax."
            )

        logger.debug(f"Resolved ${{{var_name}}} from environment")
        return env_value

    return _DEFERRED_ENV_VAR_PATTERN.sub(replacer, value)


class ComponentInstantiator:
    """Instantiates adapters and policies from native YAML dict specifications."""

    def __init__(self) -> None:
        """Initialize component instantiator."""
        pass  # No bootstrap needed anymore

    def _parse_component_spec(self, spec: dict[str, Any]) -> ComponentSpec:
        """Parse component specification from native YAML dict format.

        Parameters
        ----------
        spec : dict[str, Any]
            Component specification with either:
            - "adapter": "module.path.ClassName" and optional "config": {...}
            - "name": "module.path.ClassName" and optional "params": {...}

        Returns
        -------
        ComponentSpec
            Parsed component specification with module_path and params

        Raises
        ------
        ComponentInstantiationError
            If specification format is invalid
        """
        if not isinstance(spec, dict):
            raise ComponentInstantiationError(
                f"Component specification must be a dict, got {type(spec).__name__}. "
                f"Use format: {{adapter: 'module.path.Class', config: {{...}}}}"
            )

        # Support both "adapter" and "name" keys for module path
        module_path = spec.get("adapter") or spec.get("name")
        # Support both "config" and "params" keys for parameters
        params = spec.get("config") or spec.get("params", {})

        if not module_path:
            raise ComponentInstantiationError(
                f"Component specification requires 'adapter' or 'name' field. Got: {spec}"
            )

        return ComponentSpec(module_path=module_path, params=params)

    def instantiate_adapter(self, spec: dict[str, Any], port_name: str | None = None) -> Any:
        """Instantiate an adapter from native YAML dict specification.

        Parameters
        ----------
        spec : dict[str, Any]
            Adapter specification: {"adapter": "hexdag.stdlib.adapters.mock.MockLLM",
            "config": {"model": "gpt-4"}}
        port_name : str | None
            Optional port name for context in error messages

        Returns
        -------
        Any
            Instantiated adapter instance

        Raises
        ------
        ComponentInstantiationError
            If adapter cannot be instantiated

        Examples
        --------
        >>> instantiator = ComponentInstantiator()  # doctest: +SKIP
        >>> adapter = instantiator.instantiate_adapter({  # doctest: +SKIP
        ...     "adapter": "hexdag.stdlib.adapters.mock.MockLLM",
        ...     "config": {"model": "gpt-4"}
        ... })
        """
        try:
            # Parse the specification
            component_spec = self._parse_component_spec(spec)

            try:
                adapter_class = resolve(component_spec.module_path)
            except Exception as e:
                raise ComponentInstantiationError(
                    f"Adapter '{component_spec.module_path}' "
                    f"could not be resolved. "
                    f"Make sure the module path is correct. Error: {e}"
                ) from e

            # Instantiate the adapter class with parameters
            if isinstance(adapter_class, type):
                try:
                    # Resolve any deferred environment variables at instantiation time
                    resolved_params = _resolve_deferred_env_vars(component_spec.params)
                    adapter_instance = adapter_class(**resolved_params)
                    logger.info(
                        f"Instantiated adapter '{component_spec.module_path}' "
                        f"for port '{port_name}'"
                    )
                    return adapter_instance
                except Exception as e:
                    raise ComponentInstantiationError(
                        f"Failed to instantiate adapter "
                        f"'{component_spec.module_path}' "
                        f"with params {component_spec.params}. Error: {e}"
                    ) from e
            else:
                # It's already an instance (runtime-registered non-class component)
                if component_spec.params:  # type: ignore[unreachable]
                    logger.warning(
                        f"Adapter '{component_spec.module_path}' "
                        f"resolved to an instance. Parameters {component_spec.params} "
                        f"will be ignored."
                    )
                logger.info(
                    f"Using resolved adapter instance "
                    f"'{component_spec.module_path}' "
                    f"for port '{port_name}'"
                )
                return adapter_class

        except ComponentInstantiationError:
            raise
        except Exception as e:
            raise ComponentInstantiationError(
                f"Failed to instantiate adapter for port '{port_name}': {e}"
            ) from e

    def instantiate_policy(self, spec: dict[str, Any], policy_name: str | None = None) -> Any:
        """Instantiate a policy from native YAML dict specification.

        Parameters
        ----------
        spec : dict[str, Any]
            Policy specification: {"name": "hexdag.stdlib.policies.RetryPolicy",
            "params": {"max_retries": 3}}
        policy_name : str | None
            Optional policy name for context in error messages

        Returns
        -------
        Any
            Instantiated policy instance

        Raises
        ------
        ComponentInstantiationError
            If policy cannot be instantiated

        Examples
        --------
        >>> instantiator = ComponentInstantiator()  # doctest: +SKIP
        >>> policy = instantiator.instantiate_policy({  # doctest: +SKIP
        ...     "name": "hexdag.stdlib.policies.execution_policies.RetryPolicy",
        ...     "params": {"max_retries": 5}
        ... })
        """
        try:
            # Parse the specification
            component_spec = self._parse_component_spec(spec)

            try:
                policy_class = resolve(component_spec.module_path)
            except Exception as e:
                raise ComponentInstantiationError(
                    f"Policy '{component_spec.module_path}' "
                    f"could not be resolved. "
                    f"Make sure the module path is correct. Error: {e}"
                ) from e

            if not isinstance(policy_class, type):
                raise ComponentInstantiationError(
                    f"Policy '{component_spec.module_path}' resolved to an instance, "
                    f"not a class. Cannot instantiate from instance."
                )

            # Instantiate with parameters
            try:
                # Resolve any deferred environment variables at instantiation time
                resolved_params = _resolve_deferred_env_vars(component_spec.params)
                policy_instance = policy_class(**resolved_params)
                logger.info(f"Instantiated policy '{component_spec.module_path}' ('{policy_name}')")
                return policy_instance
            except Exception as e:
                raise ComponentInstantiationError(
                    f"Failed to instantiate policy "
                    f"'{component_spec.module_path}' "
                    f"with params {component_spec.params}. Error: {e}"
                ) from e

        except ComponentInstantiationError:
            raise
        except Exception as e:
            raise ComponentInstantiationError(
                f"Failed to instantiate policy '{policy_name}': {e}"
            ) from e

    def instantiate_ports(self, ports_config: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Instantiate all ports (adapters) from configuration.

        Parameters
        ----------
        ports_config : dict[str, dict[str, Any]]
            Dictionary of port_name -> adapter dict spec

        Returns
        -------
        dict[str, Any]
            Dictionary of port_name -> adapter_instance

        Examples
        --------
        >>> instantiator = ComponentInstantiator()  # doctest: +SKIP
        >>> config = {  # doctest: +SKIP
        ...     "llm": {"adapter": "hexdag.stdlib.adapters.mock.MockLLM",
        ...             "config": {"model": "gpt-4"}},
        ...     "database": {"adapter": "hexdag.stdlib.adapters.mock.MockDatabaseAdapter",
        ...                  "config": {}}
        ... }
        """
        ports: dict[str, Any] = {}

        for port_name, adapter_spec in ports_config.items():
            try:
                ports[port_name] = self.instantiate_adapter(adapter_spec, port_name=port_name)
            except ComponentInstantiationError as e:
                logger.error(f"Failed to instantiate adapter for port '{port_name}': {e}")
                raise

        return ports

    def instantiate_policies(self, policies_config: dict[str, dict[str, Any]]) -> list[Any]:
        """Instantiate all policies from configuration.

        Parameters
        ----------
        policies_config : dict[str, dict[str, Any]]
            Dictionary of policy_name -> policy dict spec

        Returns
        -------
        list[Any]
            List of policy instances

        Examples
        --------
        >>> instantiator = ComponentInstantiator()  # doctest: +SKIP
        >>> config = {  # doctest: +SKIP
        ...     "retry": {"name": "hexdag.stdlib.policies.RetryPolicy",
        ...               "params": {"max_retries": 3}},
        ...     "timeout": {"name": "hexdag.stdlib.policies.TimeoutPolicy",
        ...                 "params": {"timeout_seconds": 300}}
        ... }
        """
        policies: list[Any] = []

        for policy_name, policy_spec in policies_config.items():
            try:
                policy = self.instantiate_policy(policy_spec, policy_name=policy_name)
                policies.append(policy)
            except ComponentInstantiationError as e:
                logger.error(f"Failed to instantiate policy '{policy_name}': {e}")
                raise

        return policies
