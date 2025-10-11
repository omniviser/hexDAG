"""Component instantiator for creating adapters and policies from YAML specs.

This module handles:
- Parsing native YAML dict component specifications
- Resolving components from registry
- Instantiating adapters with configuration
- Instantiating policies with configuration
"""

from collections import namedtuple
from typing import Any

from hexdag.core.bootstrap import ensure_bootstrapped
from hexdag.core.logging import get_logger
from hexdag.core.registry import registry
from hexdag.core.registry.models import ClassComponent, ComponentType

logger = get_logger(__name__)

# Simple namedtuple for component specification
ComponentSpec = namedtuple("ComponentSpec", ["namespace", "name", "params"])


class ComponentInstantiationError(Exception):
    """Error instantiating component from specification."""

    pass


class ComponentInstantiator:
    """Instantiates adapters and policies from native YAML dict specifications."""

    def __init__(self) -> None:
        """Initialize component instantiator."""
        ensure_bootstrapped()

    def _parse_component_spec(self, spec: dict[str, Any]) -> ComponentSpec:
        """Parse component specification from native YAML dict format.

        Parameters
        ----------
        spec : dict[str, Any]
            Component specification: {"namespace": "ns", "name": "n", "params": {...}}

        Returns
        -------
        ComponentSpec
            Parsed component specification with namespace, name, and params

        Raises
        ------
        ComponentInstantiationError
            If specification format is invalid

        Examples
        --------
        >>> spec = self._parse_component_spec({  # doctest: +SKIP
        ...     "namespace": "core",
        ...     "name": "retry",
        ...     "params": {"max_retries": 3}
        ... })
        """
        if not isinstance(spec, dict):
            raise ComponentInstantiationError(
                f"Component specification must be a dict, got {type(spec).__name__}. "
                f"Use native YAML format: {{namespace: 'core', name: 'retry', params: {{...}}}}"
            )

        namespace = spec.get("namespace", "core")
        name = spec.get("name")
        params = spec.get("params", {})

        if not name:
            raise ComponentInstantiationError(
                f"Component specification requires 'name' field. Got: {spec}"
            )

        return ComponentSpec(namespace=namespace, name=name, params=params)

    def instantiate_adapter(self, spec: dict[str, Any], port_name: str | None = None) -> Any:
        """Instantiate an adapter from native YAML dict specification.

        Parameters
        ----------
        spec : dict[str, Any]
            Adapter specification: {"namespace": "core", "name": "openai", "params":
            {"model": "gpt-4"}}
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
        ...     "namespace": "core",
        ...     "name": "openai",
        ...     "params": {"model": "gpt-4"}
        ... })
        """
        try:
            # Parse the specification
            component_spec = self._parse_component_spec(spec)

            # Get adapter from registry
            try:
                adapter_or_class = registry.get(
                    component_spec.name, namespace=component_spec.namespace
                )
            except Exception as e:
                raise ComponentInstantiationError(
                    f"Adapter '{component_spec.namespace}:{component_spec.name}' "
                    f"not found in registry. "
                    f"Make sure it's registered in pyproject.toml. Error: {e}"
                ) from e

            # Check if we got a class or an instance
            # Some adapters (like LocalObserverManager) are registered as instances
            if isinstance(adapter_or_class, type):
                # It's a class, instantiate it with parameters
                try:
                    adapter_instance = adapter_or_class(**component_spec.params)
                    logger.info(
                        f"Instantiated adapter {component_spec.namespace}:{component_spec.name} "
                        f"for port '{port_name}' with params: {component_spec.params}"
                    )
                    return adapter_instance
                except Exception as e:
                    raise ComponentInstantiationError(
                        f"Failed to instantiate adapter "
                        f"'{component_spec.namespace}:{component_spec.name}' "
                        f"with params {component_spec.params}. Error: {e}"
                    ) from e
            else:
                # It's already an instance, check if params were provided
                if component_spec.params:
                    logger.warning(
                        f"Adapter '{component_spec.namespace}:{component_spec.name}' "
                        f"is registered as an instance. Parameters {component_spec.params} "
                        f"will be ignored."
                    )
                logger.info(
                    f"Using registered adapter instance "
                    f"{component_spec.namespace}:{component_spec.name} "
                    f"for port '{port_name}'"
                )
                return adapter_or_class

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
            Policy specification: {"namespace": "core", "name": "retry", "params":
            {"max_retries": 3}}
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
        ...     "namespace": "core",
        ...     "name": "retry",
        ...     "params": {"max_retries": 5}
        ... })
        """
        try:
            # Parse the specification
            component_spec = self._parse_component_spec(spec)

            # Get policy class from registry (policies added to bootstrap in pyproject.toml)
            policy_class = self._get_policy_class(component_spec.namespace, component_spec.name)

            # Instantiate with parameters
            try:
                policy_instance = policy_class(**component_spec.params)
                logger.info(
                    f"Instantiated policy {component_spec.namespace}:{component_spec.name} "
                    f"('{policy_name}') with params: {component_spec.params}"
                )
                return policy_instance
            except Exception as e:
                raise ComponentInstantiationError(
                    f"Failed to instantiate policy "
                    f"'{component_spec.namespace}:{component_spec.name}' "
                    f"with params {component_spec.params}. Error: {e}"
                ) from e

        except ComponentInstantiationError:
            raise
        except Exception as e:
            raise ComponentInstantiationError(
                f"Failed to instantiate policy '{policy_name}': {e}"
            ) from e

    def _get_policy_class(self, namespace: str, name: str) -> type:
        """Get policy class by namespace and name from registry.

        Parameters
        ----------
        namespace : str
            Policy namespace (e.g., "core")
        name : str
            Policy name (e.g., "retry")

        Returns
        -------
        type
            Policy class

        Raises
        ------
        ComponentInstantiationError
            If policy not found in registry
        """
        try:
            # Use get_metadata() to get the class, not get() which instantiates
            metadata = registry.get_metadata(
                name, namespace=namespace, component_type=ComponentType.POLICY
            )
            component = metadata.component

            # Extract the actual class from the ClassComponent wrapper
            policy_class = component.value if isinstance(component, ClassComponent) else component

            if not isinstance(policy_class, type):
                raise ComponentInstantiationError(
                    f"Policy '{namespace}:{name}' is registered as an instance, not a class. "
                    f"Cannot instantiate from instance."
                )

            return policy_class
        except Exception as e:
            raise ComponentInstantiationError(
                f"Policy '{namespace}:{name}' not found in registry. "
                f"Ensure policies are registered via @policy decorator "
                f"and the policy module is in bootstrap config. Error: {e}"
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
        ...     "llm": {"namespace": "core", "name": "openai", "params": {"model": "gpt-4"}},
        ...     "database": {"namespace": "core", "name": "postgres", "params":
                {"connection_string": "..."}}
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
        ...     "retry": {"name": "retry", "params": {"max_retries": 3}},
        ...     "timeout": {"name": "timeout", "params": {"timeout_seconds": 300}}
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
