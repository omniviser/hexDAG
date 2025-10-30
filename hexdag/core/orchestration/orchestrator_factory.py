"""Orchestrator Factory - Creates orchestrator instances from pipeline configuration.

This factory bridges the gap between declarative YAML configuration and the
runtime orchestrator, instantiating adapters and policies from their specs.
"""

from typing import Any

from hexdag.core.logging import get_logger
from hexdag.core.orchestration.orchestrator import Orchestrator
from hexdag.core.pipeline_builder.component_instantiator import ComponentInstantiator
from hexdag.core.pipeline_builder.pipeline_config import PipelineConfig
from hexdag.core.ports_builder import PortsBuilder

logger = get_logger(__name__)


class OrchestratorFactory:
    """Factory for creating orchestrator instances from pipeline configuration.

    This factory handles the complex task of:
    1. Instantiating adapter instances from port specs
    2. Instantiating policy instances from policy specs
    3. Wiring everything together into a configured orchestrator

    Examples
    --------
    ```python
    from hexdag.core.pipeline_builder.yaml_builder import YamlPipelineBuilder
    from hexdag.core.orchestration.orchestrator_factory import OrchestratorFactory

    # Parse YAML pipeline
    builder = YamlPipelineBuilder()
    graph, pipeline_config = builder.build_from_yaml_file("pipeline.yaml")

    factory = OrchestratorFactory()
    orchestrator = factory.create_orchestrator(pipeline_config)

    # Execute the pipeline
    result = await orchestrator.aexecute_dag(graph, initial_input={"query": "..."})
    ```
    """

    def __init__(self) -> None:
        """Initialize the orchestrator factory."""
        self.component_instantiator = ComponentInstantiator()

    def create_orchestrator(
        self,
        pipeline_config: PipelineConfig,
        max_concurrent_nodes: int = 10,
        strict_validation: bool = False,
        default_node_timeout: float | None = None,
        additional_ports: dict[str, Any] | None = None,
    ) -> Orchestrator:
        """Create an orchestrator instance from pipeline configuration.

        Parameters
        ----------
        pipeline_config : PipelineConfig
            Pipeline configuration with ports, policies, and metadata
        max_concurrent_nodes : int, optional
            Maximum number of nodes to execute concurrently, by default 10
        strict_validation : bool, optional
            If True, raise errors on validation failure, by default False
        default_node_timeout : float | None, optional
            Default timeout in seconds for each node, by default None
        additional_ports : dict[str, Any] | None, optional
            Additional ports to merge with configured ports, by default None

        Returns
        -------
        Orchestrator
            Configured orchestrator ready to execute pipelines

        Examples
        --------
        Basic usage with global ports::

            factory = OrchestratorFactory()

            orchestrator = factory.create_orchestrator(
                pipeline_config=config,
                max_concurrent_nodes=5,
                strict_validation=True,
                default_node_timeout=60.0,
            )

        Notes
        -----
        **Known Limitations (Phase 4)**:

        - **Type-specific ports** (``type_ports`` in YAML): Parsed and validated
          but not yet used during execution. All nodes currently receive global ports.

        - **Node-level port overrides**: Not yet implemented. Requires orchestrator
          changes to resolve ports per-node at execution time.

        **Workaround for Advanced Port Configuration**:

        For type-specific or node-level port customization, use PortsBuilder
        programmatically::

            from hexdag.core.ports_builder import PortsBuilder

            builder = (
                PortsBuilder()
                .with_llm(MockLLM())  # Global default
                .for_type("agent", llm=OpenAIAdapter(model="gpt-4"))
                .for_node("researcher", llm=AnthropicAdapter(model="claude-3"))
            )

            # Pass as additional_ports
            orchestrator = factory.create_orchestrator(
                pipeline_config,
                additional_ports=builder.build()
            )

        These features are planned for future implementation when orchestrator
        supports per-node port resolution.
        """
        logger.info(
            "Creating orchestrator from pipeline config: {} ports, {} type_ports, {} policies",
            len(pipeline_config.ports),
            len(pipeline_config.type_ports),
            len(pipeline_config.policies),
        )

        # Step 1: Build PortsConfiguration if type_ports or node_ports are configured
        use_ports_config = bool(pipeline_config.type_ports)

        if use_ports_config:
            ports_config = self._build_ports_configuration(pipeline_config, additional_ports)
            # Note: PortsConfiguration converts dicts to tuples for immutability
            global_ports = {k: v.port for k, v in (ports_config.global_ports or ())}
        else:
            # Simple case: just global ports
            ports_config = None
            global_ports = self._instantiate_ports(pipeline_config.ports)
            if additional_ports:
                global_ports.update(additional_ports)

        # Step 2: Instantiate policies from policy specs
        policies = self._instantiate_policies(pipeline_config.policies)

        # Step 3: Subscribe policies to policy manager if available
        if "policy_manager" in global_ports and policies:
            self._subscribe_policies_to_manager(global_ports["policy_manager"], policies)

        # Step 4: Create orchestrator with configured ports
        orchestrator = Orchestrator(
            max_concurrent_nodes=max_concurrent_nodes,
            ports=ports_config if ports_config else global_ports,
            strict_validation=strict_validation,
            default_node_timeout=default_node_timeout,
        )

        logger.info(
            "✅ Orchestrator created with {} ports and {} policies",
            len(global_ports),
            len(policies),
        )

        return orchestrator

    def _instantiate_ports(self, port_specs: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Instantiate adapter instances from port specifications.

        Parameters
        ----------
        port_specs : dict[str, dict[str, Any]]
            Map of port_name -> adapter dict spec

        Returns
        -------
        dict[str, Any]
            Map of port_name -> adapter_instance
        """
        ports = {}

        for port_name, port_spec in port_specs.items():
            try:
                logger.debug("Instantiating port: {} = {}", port_name, port_spec)
                adapter = self.component_instantiator.instantiate_adapter(port_spec, port_name)
                ports[port_name] = adapter
                logger.debug("✅ Port instantiated: {}", port_name)
            except Exception as e:
                logger.error(
                    "Failed to instantiate port '{}' from spec '{}': {}",
                    port_name,
                    port_spec,
                    e,
                )
                raise

        return ports

    def _instantiate_policies(self, policy_specs: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Instantiate policy instances from policy specifications.

        Parameters
        ----------
        policy_specs : dict[str, dict[str, Any]]
            Map of policy_name -> policy dict spec

        Returns
        -------
        dict[str, Any]
            Map of policy_name -> policy_instance
        """
        policies = {}

        for policy_name, policy_spec in policy_specs.items():
            try:
                logger.debug("Instantiating policy: {} = {}", policy_name, policy_spec)
                policy = self.component_instantiator.instantiate_policy(policy_spec, policy_name)
                policies[policy_name] = policy
                logger.debug("✅ Policy instantiated: {}", policy_name)
            except Exception as e:
                logger.error(
                    "Failed to instantiate policy '{}' from spec '{}': {}",
                    policy_name,
                    policy_spec,
                    e,
                )
                raise

        return policies

    def _subscribe_policies_to_manager(self, policy_manager: Any, policies: dict[str, Any]) -> None:
        """Subscribe policy instances to the policy manager.

        Parameters
        ----------
        policy_manager : Any
            Policy manager port instance
        policies : dict[str, Any]
            Map of policy_name -> policy_instance
        """
        if not hasattr(policy_manager, "subscribe"):
            logger.warning(
                "Policy manager does not have 'subscribe' method, skipping policy subscription"
            )
            return

        for policy_name, policy in policies.items():
            try:
                policy_manager.subscribe(policy)
                logger.debug("Subscribed policy '{}' to policy manager", policy_name)
            except Exception as e:
                logger.warning("Failed to subscribe policy '{}': {}", policy_name, e)

    def _build_ports_configuration(
        self, pipeline_config: PipelineConfig, additional_ports: dict[str, Any] | None
    ) -> Any:
        """Build a PortsConfiguration from pipeline config.

        Parameters
        ----------
        pipeline_config : PipelineConfig
            Pipeline configuration with ports, type_ports
        additional_ports : dict[str, Any] | None
            Additional ports to merge

        Returns
        -------
        PortsConfiguration
            Hierarchical ports configuration with type-specific support
        """
        from hexdag.core.orchestration.models import PortConfig, PortsConfiguration

        # Step 1: Instantiate global ports
        global_ports_dict = self._instantiate_ports(pipeline_config.ports)
        if additional_ports:
            global_ports_dict.update(additional_ports)

        # Wrap in PortConfig
        global_ports = {k: PortConfig(port=v) for k, v in global_ports_dict.items()}

        # Step 2: Instantiate type-specific ports
        type_ports = None
        if pipeline_config.type_ports:
            type_ports = {}
            for node_type, type_port_specs in pipeline_config.type_ports.items():
                type_port_instances = self._instantiate_ports(type_port_specs)
                type_ports[node_type] = {
                    k: PortConfig(port=v) for k, v in type_port_instances.items()
                }
                logger.debug(
                    "Configured {} ports for node type '{}'",
                    len(type_port_instances),
                    node_type,
                )

        # Step 3: TODO: Add node-level port support when available in YAML builder

        return PortsConfiguration(
            global_ports=global_ports,
            type_ports=type_ports,
            node_ports=None,  # Not yet implemented
        )

    def create_ports_builder(self, pipeline_config: PipelineConfig) -> PortsBuilder:
        """Create a PortsBuilder from pipeline configuration (convenience method).

        This method provides an alternative workflow using PortsBuilder for
        advanced use cases that need type-specific or node-specific port overrides.

        Parameters
        ----------
        pipeline_config : PipelineConfig
            Pipeline configuration with ports

        Returns
        -------
        PortsBuilder
            Builder with ports configured from pipeline config

        Examples
        --------
        ```python
        factory = OrchestratorFactory()

        builder = factory.create_ports_builder(pipeline_config)

        # Optionally add node-specific overrides
        builder.for_node("researcher", llm=custom_llm)

        ports = builder.build()
        ```
        """
        ports = self._instantiate_ports(pipeline_config.ports)

        builder = PortsBuilder()
        for port_name, port_instance in ports.items():
            builder.with_custom(port_name, port_instance)

        return builder
