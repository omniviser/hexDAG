"""Component instantiator for creating adapters and policies from YAML specs.

This module handles:
- Parsing native YAML dict component specifications
- Resolving components via module paths
- Instantiating adapters with configuration
- Instantiating policies with configuration
"""

from collections import namedtuple
from typing import Any

from hexdag.core.logging import get_logger
from hexdag.core.resolver import resolve

logger = get_logger(__name__)

# Simple namedtuple for component specification
ComponentSpec = namedtuple("ComponentSpec", ["module_path", "params"])


class ComponentInstantiationError(Exception):
    """Error instantiating component from specification."""

    pass


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
            Adapter specification: {"adapter": "hexdag.builtin.adapters.mock.MockLLM",
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
        ...     "adapter": "hexdag.builtin.adapters.mock.MockLLM",
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
                    adapter_instance = adapter_class(**component_spec.params)
                    logger.info(
                        f"Instantiated adapter '{component_spec.module_path}' "
                        f"for port '{port_name}' with params: {component_spec.params}"
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
            Policy specification: {"name": "hexdag.builtin.policies.RetryPolicy",
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
        ...     "name": "hexdag.builtin.policies.execution_policies.RetryPolicy",
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
                policy_instance = policy_class(**component_spec.params)
                logger.info(
                    f"Instantiated policy '{component_spec.module_path}' "
                    f"('{policy_name}') with params: {component_spec.params}"
                )
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
        ...     "llm": {"adapter": "hexdag.builtin.adapters.mock.MockLLM",
        ...             "config": {"model": "gpt-4"}},
        ...     "database": {"adapter": "hexdag.builtin.adapters.mock.MockDatabaseAdapter",
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
        ...     "retry": {"name": "hexdag.builtin.policies.RetryPolicy",
        ...               "params": {"max_retries": 3}},
        ...     "timeout": {"name": "hexdag.builtin.policies.TimeoutPolicy",
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
