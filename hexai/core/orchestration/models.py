"""Models for orchestration state and checkpoints.

This module contains models for representing orchestrator execution
state, execution context, and human-in-the-loop approval requests.
It also includes port configuration models for managing per-node and
per-type port customization.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from hexai.core.exceptions import ValidationError


@dataclass(frozen=True, slots=True)
class OrchestratorConfig:
    """Configuration for orchestrator behavior.

    This immutable configuration object centralizes all orchestrator settings,
    making it easier to pass configurations around and test different settings.

    Attributes
    ----------
    max_concurrent_nodes : int, default=10
        Maximum number of nodes to execute concurrently in a wave.
        Controls parallelism and resource usage.
    strict_validation : bool, default=False
        If True, raise errors on validation failures.
        If False, log warnings and continue execution.
    default_node_timeout : float | None, default=None
        Default timeout in seconds for node execution.
        None means no timeout. Can be overridden per-node.

    Examples
    --------
    Example usage::

        config = OrchestratorConfig(
        max_concurrent_nodes=5,
        strict_validation=True,
        default_node_timeout=30.0
        )
        orchestrator = Orchestrator(config=config)

        # Or use defaults
        config = OrchestratorConfig()
        config.max_concurrent_nodes
    10
    """

    max_concurrent_nodes: int = 10
    strict_validation: bool = False
    default_node_timeout: float | None = None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.max_concurrent_nodes <= 0:
            raise ValidationError(
                "max_concurrent_nodes", "must be positive", self.max_concurrent_nodes
            )

        if self.default_node_timeout is not None and self.default_node_timeout <= 0:
            raise ValidationError(
                "default_node_timeout", "must be positive or None", self.default_node_timeout
            )


class CheckpointState(BaseModel):
    """Complete state for checkpoint/resume.

    Saves everything needed to resume a DAG execution from where it left off,
    including the full graph structure (handles both static and dynamic DAGs).

    Attributes
    ----------
    run_id : str
        Unique identifier for this execution run
    dag_id : str
        Stable identifier for the DAG (e.g., YAML file path, function name)
    graph_snapshot : dict[str, Any]
        Serialized DirectedGraph structure
        Format: {"nodes": {...}, "edges": [...]}
    initial_input : Any
        Initial input data passed to the DAG
    node_results : dict[str, Any]
        Results from completed nodes (node_id -> output)
    completed_node_ids : list[str]
        Ordered list of completed node IDs (preserves execution order)
    failed_node_ids : list[str]
        List of node IDs that failed (for retry/debugging)
    created_at : datetime
        When execution started
    updated_at : datetime
        Last checkpoint save time
    metadata : dict[str, Any]
        Optional metadata (custom fields, tags, etc.)

    Notes
    -----
    To resume execution:
    1. Load CheckpointState by run_id
    2. Deserialize graph_snapshot to DirectedGraph
    3. Filter out completed nodes using completed_node_ids
    4. Resume execution with filtered graph and node_results
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    dag_id: str
    graph_snapshot: dict[str, Any]  # Always save the graph
    initial_input: Any
    node_results: dict[str, Any]  # node_id -> output
    completed_node_ids: list[str]  # Ordered
    failed_node_ids: list[str] = []
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = {}


@dataclass(slots=True)
class NodeExecutionContext:
    """Lightweight context tracking current execution position.

    This flows through the execution pipeline and is NOT persisted in checkpoints.
    It's only for tracking where we are during live execution (which node, wave, attempt).

    Attributes
    ----------
    dag_id : str
        Identifier for the DAG being executed
    node_id : str | None
        Current node being executed (None for DAG-level operations)
    wave_index : int
        Index of the current execution wave (for parallel execution tracking)
    attempt : int
        Attempt number (for retry scenarios)
    metadata : dict[str, Any]
        Additional metadata that can be attached to the context
    """

    dag_id: str
    node_id: str | None = None
    wave_index: int = 0
    attempt: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_node(self, node_id: str, wave_index: int) -> "NodeExecutionContext":
        """Create new context for a specific node execution.

        Args
        ----
            node_id: The ID of the node being executed
            wave_index: The wave index for parallel execution tracking

        Returns
        -------
        NodeExecutionContext
            New context with updated node and wave information
        """
        return NodeExecutionContext(
            dag_id=self.dag_id,
            node_id=node_id,
            wave_index=wave_index,
            attempt=self.attempt,
            metadata=self.metadata.copy(),
        )

    def with_attempt(self, attempt: int) -> "NodeExecutionContext":
        """Create new context with updated attempt number.

        Args
        ----
            attempt: The attempt number (for retry scenarios)

        Returns
        -------
        NodeExecutionContext
            New context with updated attempt number
        """
        return NodeExecutionContext(
            dag_id=self.dag_id,
            node_id=self.node_id,
            wave_index=self.wave_index,
            attempt=attempt,
            metadata=self.metadata.copy(),
        )


@dataclass(frozen=True, slots=True)
class PortConfig:
    """Configuration for a single port instance.

    A PortConfig wraps a port implementation with optional metadata,
    allowing for fine-grained control over port behavior per node.

    Attributes
    ----------
    port : Any
        The port implementation instance (e.g., LLM, Database, Memory adapter)
    metadata : Mapping[str, Any] | None
        Optional metadata for the port (e.g., timeouts, retry settings, rate limits)

    Examples
    --------
    Example usage::

        from hexai.adapters.mock import MockLLM
        config = PortConfig(
        port=MockLLM(),
        metadata={"timeout": 30, "max_retries": 3}
        )
        config.port
    <MockLLM object>
        config.get_metadata()
    {'timeout': 30, 'max_retries': 3}
    """

    port: Any
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        """Ensure metadata is immutable if provided."""
        if self.metadata is not None:
            # Convert to tuple of items for immutability
            object.__setattr__(self, "metadata", tuple(self.metadata.items()))

    def get_metadata(self) -> dict[str, Any]:
        """Get metadata as a dictionary.

        Returns
        -------
        dict[str, Any]
            Metadata dictionary (empty if no metadata)
        """
        if self.metadata is None:
            return {}
        return dict(self.metadata)


@dataclass(frozen=True, slots=True)
class PortsConfiguration:
    """Complete port configuration with inheritance and overrides.

    This model supports three levels of port configuration with clear inheritance:
    1. **Global defaults** - Apply to all nodes unless overridden
    2. **Per-type defaults** - Apply to all nodes of a specific type (e.g., "agent", "llm")
    3. **Per-node overrides** - Apply to specific nodes by name

    **Resolution order**: per-node > per-type > global defaults

    Attributes
    ----------
    global_ports : Mapping[str, PortConfig] | None
        Default ports for all nodes
    type_ports : Mapping[str, Mapping[str, PortConfig]] | None
        Ports per node type, keyed by type name (e.g., {"agent": {"llm": config}})
    node_ports : Mapping[str, Mapping[str, PortConfig]] | None
        Ports per specific node name (e.g., {"researcher": {"llm": config}})

    Examples
    --------
    Example usage::

        from hexai.adapters.mock import MockLLM
        from hexai.adapters.openai import OpenAIAdapter
        from hexai.adapters.anthropic import AnthropicAdapter

        # Global default: All nodes use MockLLM
        config = PortsConfiguration(
        global_ports={"llm": PortConfig(MockLLM())},
        type_ports={
        # Override for all "agent" type nodes
        "agent": {"llm": PortConfig(OpenAIAdapter(model="gpt-4"))}
        },
        node_ports={
        # Override for specific "researcher" node
        "researcher": {"llm": PortConfig(AnthropicAdapter(model="claude-3"))}
        }
        )

        # Resolution for different nodes:
        # - "researcher" node: AnthropicAdapter (per-node override)
        researcher_ports = config.resolve_ports("researcher", "agent")
        assert isinstance(researcher_ports["llm"].port, AnthropicAdapter)

        # - Other "agent" nodes: OpenAIAdapter (per-type default)
        agent_ports = config.resolve_ports("analyzer", "agent")
        assert isinstance(agent_ports["llm"].port, OpenAIAdapter)

        # - Other nodes: MockLLM (global default)
        function_ports = config.resolve_ports("transformer", "function")
        assert isinstance(function_ports["llm"].port, MockLLM)

    Notes
    -----
    This design enables:
    - **Cost optimization**: Use cheaper models for simple nodes, expensive ones for complex tasks
    - **Performance tuning**: Different timeout/retry settings per node type
    - **Testing flexibility**: Mock some nodes, use real adapters for others
    - **Multi-tenant support**: Different credentials per node/type
    """

    global_ports: Mapping[str, PortConfig] | None = None
    type_ports: Mapping[str, Mapping[str, PortConfig]] | None = None
    node_ports: Mapping[str, Mapping[str, PortConfig]] | None = None

    def __post_init__(self) -> None:
        """Ensure all mappings are immutable."""
        if self.global_ports is not None:
            object.__setattr__(self, "global_ports", tuple(self.global_ports.items()))
        if self.type_ports is not None:
            # Convert nested dict to tuple of (key, tuple of items)
            type_items = tuple(
                (node_type, tuple(ports.items())) for node_type, ports in self.type_ports.items()
            )
            object.__setattr__(self, "type_ports", type_items)
        if self.node_ports is not None:
            # Convert nested dict to tuple of (key, tuple of items)
            node_items = tuple(
                (node_name, tuple(ports.items())) for node_name, ports in self.node_ports.items()
            )
            object.__setattr__(self, "node_ports", node_items)

    def resolve_ports(self, node_name: str, node_type: str | None = None) -> dict[str, PortConfig]:
        """Resolve ports for a specific node following inheritance rules.

        Combines ports from all three levels (global, type, node) with proper
        precedence: per-node > per-type > global defaults.

        Parameters
        ----------
        node_name : str
            Name of the node to resolve ports for
        node_type : str | None
            Type of the node (e.g., "llm", "agent", "function", "loop")

        Returns
        -------
        dict[str, PortConfig]
            Resolved ports for the node with PortConfig wrappers

        Examples
        --------
        Example usage::

            config = PortsConfiguration(
            global_ports={"llm": PortConfig(MockLLM())},
            type_ports={"agent": {"llm": PortConfig(OpenAIAdapter())}},
            node_ports={"researcher": {"llm": PortConfig(AnthropicAdapter())}}
            )

            # Researcher node gets Anthropic (per-node override)
            researcher_ports = config.resolve_ports("researcher", "agent")
            assert isinstance(researcher_ports["llm"].port, AnthropicAdapter)

            # Other agent nodes get OpenAI (per-type default)
            agent_ports = config.resolve_ports("analyzer", "agent")
            assert isinstance(agent_ports["llm"].port, OpenAIAdapter)

            # Function nodes get Mock (global default)
            function_ports = config.resolve_ports("transformer", "function")
            assert isinstance(function_ports["llm"].port, MockLLM)
        """
        result: dict[str, PortConfig] = {}

        # 1. Start with global defaults (lowest priority)
        if self.global_ports is not None:
            result.update(dict(self.global_ports))

        # 2. Apply per-type defaults (overrides global)
        if self.type_ports is not None and node_type is not None:
            type_dict = dict(self.type_ports)
            if node_type in type_dict:
                result.update(dict(type_dict[node_type]))

        # 3. Apply per-node overrides (highest priority)
        if self.node_ports is not None:
            node_dict = dict(self.node_ports)
            if node_name in node_dict:
                result.update(dict(node_dict[node_name]))

        return result

    def to_flat_dict(self, node_name: str, node_type: str | None = None) -> dict[str, Any]:
        """Convert resolved ports to flat dictionary of port instances.

        This extracts the actual port instances from PortConfig wrappers,
        providing backward compatibility with the current orchestrator
        interface that expects `dict[str, Any]` ports.

        Parameters
        ----------
        node_name : str
            Name of the node
        node_type : str | None
            Type of the node

        Returns
        -------
        dict[str, Any]
            Dictionary mapping port names to port instances (unwrapped)

        Examples
        --------
        Example usage::

            config = PortsConfiguration(
            global_ports={"llm": PortConfig(MockLLM())}
            )

            # Get flat dictionary for orchestrator
            ports = config.to_flat_dict("my_node")
            assert "llm" in ports
            assert isinstance(ports["llm"], MockLLM)  # Unwrapped
        """
        resolved = self.resolve_ports(node_name, node_type)
        return {port_name: config.port for port_name, config in resolved.items()}
