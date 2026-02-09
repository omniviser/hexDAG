"""Per-pipeline environment management API for hexdag studio.

Environments support:
1. Global ports - adapters that apply to all nodes
2. Per-node overrides - specific adapter configurations for individual nodes
3. Adapter config schemas - dynamic form generation from adapter metadata

Environments are discovered per-pipeline:
1. From `environments/` subfolder (e.g., environments/local.yaml)
2. From inline `spec.environments` in the pipeline YAML
3. Default mock environment when nothing configured
"""

import os
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from hexdag_studio.server.routes.files import get_workspace_root

router = APIRouter(prefix="/environments", tags=["environments"])


def _resolve_path(relative_path: str) -> Path:
    """Resolve a relative path within the workspace.

    Prevents directory traversal attacks.
    """
    root = get_workspace_root()
    resolved = (root / relative_path).resolve()

    # Security: ensure path is within workspace
    if not str(resolved).startswith(str(root.resolve())):
        raise HTTPException(status_code=403, detail="Access denied: path outside workspace")

    return resolved


class PortConfig(BaseModel):
    """Configuration for a single port."""

    adapter: str
    config: dict[str, Any] = {}


class EnvironmentConfig(BaseModel):
    """Configuration for an environment.

    Supports global ports and per-node overrides.
    """

    name: str
    description: str = ""
    ports: dict[str, PortConfig] = {}  # Global ports for all nodes
    node_overrides: dict[str, dict[str, PortConfig]] = {}  # node_name -> port_type -> config


class EnvironmentsResponse(BaseModel):
    """List of available environments for a pipeline."""

    environments: list[EnvironmentConfig]
    current: str
    source: str  # 'folder', 'inline', or 'default'


class AdapterInfo(BaseModel):
    """Information about an available adapter."""

    name: str
    port_type: str
    description: str
    config_schema: dict[str, Any]
    secrets: list[str]


def get_default_environment() -> EnvironmentConfig:
    """Get the default mock environment for testing."""
    return EnvironmentConfig(
        name="local",
        description="Local mock environment for testing",
        ports={
            "llm": PortConfig(adapter="MockLLM", config={}),
            "memory": PortConfig(adapter="InMemoryMemory", config={}),
        },
    )


def load_environment_from_file(env_file: Path) -> EnvironmentConfig | None:
    """Load an environment configuration from a YAML file."""
    try:
        with Path(env_file).open() as f:
            data = yaml.safe_load(f)

        if not data:
            return None

        name = env_file.stem  # filename without extension

        # Parse global ports
        ports: dict[str, PortConfig] = {}
        for port_name, port_data in data.get("ports", {}).items():
            if isinstance(port_data, dict):
                ports[port_name] = PortConfig(
                    adapter=port_data.get("adapter", ""),
                    config=port_data.get("config", {}),
                )

        # Parse per-node overrides
        node_overrides: dict[str, dict[str, PortConfig]] = {}
        for node_name, node_ports in data.get("node_overrides", {}).items():
            if isinstance(node_ports, dict):
                node_overrides[node_name] = {}
                for port_name, port_data in node_ports.items():
                    if isinstance(port_data, dict):
                        node_overrides[node_name][port_name] = PortConfig(
                            adapter=port_data.get("adapter", ""),
                            config=port_data.get("config", {}),
                        )

        return EnvironmentConfig(
            name=name,
            description=data.get("description", ""),
            ports=ports,
            node_overrides=node_overrides,
        )
    except Exception as e:
        print(f"Failed to load environment {env_file}: {e}")
        return None


def extract_node_level_ports(yaml_content: str) -> EnvironmentConfig | None:
    """Extract ports defined at node level (spec.nodes[].spec.ports) into an environment.

    This handles the pattern where ports are defined inline in each node:
    ```yaml
    spec:
      nodes:
        - kind: llm_node
          metadata:
            name: analyzer_a
          spec:
            prompt_template: "..."
            ports:
              llm:
                adapter: MockLLM
                config:
                  responses: hello
    ```

    Returns an environment with node_overrides for each node that has ports.
    """
    try:
        parsed = yaml.safe_load(yaml_content)
        if not parsed or not isinstance(parsed, dict):
            return None

        spec = parsed.get("spec", {})
        nodes = spec.get("nodes", [])

        if not nodes or not isinstance(nodes, list):
            return None

        # Build node_overrides from each node's spec.ports
        node_overrides: dict[str, dict[str, PortConfig]] = {}
        has_node_ports = False

        for node in nodes:
            if not isinstance(node, dict):
                continue

            node_name = node.get("metadata", {}).get("name", "")
            node_spec = node.get("spec", {})
            node_ports = node_spec.get("ports", {})

            if not node_name or not node_ports:
                continue

            if not isinstance(node_ports, dict):
                continue

            node_overrides[node_name] = {}
            for port_name, port_data in node_ports.items():
                if isinstance(port_data, dict):
                    node_overrides[node_name][port_name] = PortConfig(
                        adapter=port_data.get("adapter", ""),
                        config=port_data.get("config", {}),
                    )
                    has_node_ports = True

        if not has_node_ports:
            return None

        # Return a synthetic "node_ports" environment
        return EnvironmentConfig(
            name="node_ports",
            description="Ports from node-level configuration",
            ports={
                # Default global ports for nodes without overrides
                "llm": PortConfig(adapter="MockLLM", config={}),
                "memory": PortConfig(adapter="InMemoryMemory", config={}),
            },
            node_overrides=node_overrides,
        )
    except Exception as e:
        print(f"Failed to extract node-level ports: {e}")
        return None


def parse_inline_environments(yaml_content: str) -> list[EnvironmentConfig]:
    """Parse environments defined inline in pipeline YAML.

    Example:
    ```yaml
    spec:
      environments:
        local:
          description: Local testing
          ports:
            llm:
              adapter: MockLLM
          node_overrides:
            analyzer_b:
              llm:
                adapter: anthropic
                config:
                  model: claude-3-sonnet
        dev:
          ports:
            llm:
              adapter: openai
              config:
                api_key: ${OPENAI_API_KEY}
    ```
    """
    try:
        parsed = yaml.safe_load(yaml_content)
        if not parsed or not isinstance(parsed, dict):
            return []

        spec = parsed.get("spec", {})
        inline_envs = spec.get("environments", {})

        if not inline_envs or not isinstance(inline_envs, dict):
            return []

        environments = []
        for env_name, env_data in inline_envs.items():
            if not isinstance(env_data, dict):
                continue

            # Parse global ports
            ports: dict[str, PortConfig] = {}
            for port_name, port_data in env_data.get("ports", {}).items():
                if isinstance(port_data, dict):
                    ports[port_name] = PortConfig(
                        adapter=port_data.get("adapter", ""),
                        config=port_data.get("config", {}),
                    )

            # Parse per-node overrides
            node_overrides: dict[str, dict[str, PortConfig]] = {}
            for node_name, node_ports in env_data.get("node_overrides", {}).items():
                if isinstance(node_ports, dict):
                    node_overrides[node_name] = {}
                    for port_name, port_data in node_ports.items():
                        if isinstance(port_data, dict):
                            node_overrides[node_name][port_name] = PortConfig(
                                adapter=port_data.get("adapter", ""),
                                config=port_data.get("config", {}),
                            )

            environments.append(
                EnvironmentConfig(
                    name=env_name,
                    description=env_data.get("description", ""),
                    ports=ports,
                    node_overrides=node_overrides,
                )
            )

        return environments
    except Exception as e:
        print(f"Failed to parse inline environments: {e}")
        return []


def discover_folder_environments(pipeline_dir: Path) -> list[EnvironmentConfig]:
    """Discover environments from a pipeline's environments/ subfolder."""
    env_dir = pipeline_dir / "environments"

    if not env_dir.exists():
        return []

    environments = []
    for env_file in sorted(env_dir.glob("*.yaml")):
        env = load_environment_from_file(env_file)
        if env:
            environments.append(env)

    return environments


def discover_environments_for_pipeline(
    pipeline_path: str | None = None,
    yaml_content: str | None = None,
) -> tuple[list[EnvironmentConfig], str]:
    """Discover environments for a specific pipeline.

    Discovery order:
    1. Node-level ports in YAML (spec.nodes[].spec.ports) - highest priority
    2. Inline environments in YAML (spec.environments)
    3. environments/ folder next to pipeline
    4. Default mock environment

    Returns (environments, source) where source is 'node_ports', 'inline', 'folder', or 'default'.
    """
    if yaml_content:
        # First check for node-level ports (spec.nodes[].spec.ports)
        node_ports_env = extract_node_level_ports(yaml_content)
        if node_ports_env:
            return [node_ports_env], "node_ports"

        # Then check inline environments (spec.environments)
        inline_envs = parse_inline_environments(yaml_content)
        if inline_envs:
            return inline_envs, "inline"

    # Then check folder-based environments
    if pipeline_path:
        full_path = _resolve_path(pipeline_path)
        # Get parent directory if this is a file path (has .yaml extension)
        if (
            pipeline_path.endswith(".yaml")
            or pipeline_path.endswith(".yml")
            or full_path.exists()
            and full_path.is_file()
        ):
            pipeline_dir = full_path.parent
        else:
            pipeline_dir = full_path

        folder_envs = discover_folder_environments(pipeline_dir)
        if folder_envs:
            return folder_envs, "folder"

    # Default to mock environment
    return [get_default_environment()], "default"


def resolve_env_vars(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve ${VAR_NAME} references in config values."""
    resolved = {}
    for key, value in config.items():
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            var_name = value[2:-1]
            resolved[key] = os.environ.get(var_name, "")
        elif isinstance(value, dict):
            resolved[key] = resolve_env_vars(value)
        else:
            resolved[key] = value
    return resolved


def get_environment_ports(
    env_name: str,
    pipeline_path: str | None = None,
    yaml_content: str | None = None,
) -> dict[str, Any]:
    """Get resolved port configurations for a specific environment."""
    environments, _ = discover_environments_for_pipeline(pipeline_path, yaml_content)

    for env in environments:
        if env.name == env_name:
            ports = {}
            for port_name, port_config in env.ports.items():
                resolved_config = resolve_env_vars(port_config.config)
                ports[port_name] = {
                    "adapter": port_config.adapter,
                    "config": resolved_config,
                }
            return ports

    return {}


def get_environment_ports_with_overrides(
    env_name: str,
    pipeline_path: str | None = None,
    yaml_content: str | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Get resolved port configurations including per-node overrides.

    Returns
    -------
    tuple[dict, dict]
        - global_ports: Dict of port_name -> {adapter, config}
        - node_overrides: Dict of node_name -> {port_type -> {adapter, config}}
    """
    environments, _ = discover_environments_for_pipeline(pipeline_path, yaml_content)

    for env in environments:
        if env.name == env_name:
            # Global ports
            global_ports: dict[str, Any] = {}
            for port_name, port_config in env.ports.items():
                resolved_config = resolve_env_vars(port_config.config)
                global_ports[port_name] = {
                    "adapter": port_config.adapter,
                    "config": resolved_config,
                }

            # Node overrides
            node_overrides: dict[str, dict[str, Any]] = {}
            for node_name, node_port_configs in env.node_overrides.items():
                node_overrides[node_name] = {}
                for port_type, port_config in node_port_configs.items():
                    resolved_config = resolve_env_vars(port_config.config)
                    node_overrides[node_name][port_type] = {
                        "adapter": port_config.adapter,
                        "config": resolved_config,
                    }

            return global_ports, node_overrides

    return {}, {}


def get_node_port_config(
    env_name: str,
    node_name: str,
    port_type: str,
    pipeline_path: str | None = None,
    yaml_content: str | None = None,
) -> dict[str, Any] | None:
    """Get the effective port config for a specific node.

    Returns node override if exists, otherwise global port config.
    """
    environments, _ = discover_environments_for_pipeline(pipeline_path, yaml_content)

    for env in environments:
        if env.name == env_name:
            # Check node override first
            if node_name in env.node_overrides and port_type in env.node_overrides[node_name]:
                port = env.node_overrides[node_name][port_type]
                return {
                    "adapter": port.adapter,
                    "config": resolve_env_vars(port.config),
                    "source": "node_override",
                }

            # Fall back to global port
            if port_type in env.ports:
                port = env.ports[port_type]
                return {
                    "adapter": port.adapter,
                    "config": resolve_env_vars(port.config),
                    "source": "global",
                }

    return None


# ===== API Endpoints =====


class DiscoverRequest(BaseModel):
    """Request to discover environments for a pipeline."""

    pipeline_path: str | None = None
    yaml_content: str | None = None


@router.post("/discover", response_model=EnvironmentsResponse)
async def discover_environments(request: DiscoverRequest) -> EnvironmentsResponse:
    """Discover environments for a specific pipeline.

    Searches in order:
    1. Inline in YAML (spec.environments)
    2. environments/ folder relative to pipeline
    3. Falls back to default mock environment
    """
    environments, source = discover_environments_for_pipeline(
        request.pipeline_path,
        request.yaml_content,
    )

    # Default to first environment as "current"
    current = environments[0].name if environments else "local"

    return EnvironmentsResponse(
        environments=environments,
        current=current,
        source=source,
    )


@router.get("", response_model=EnvironmentsResponse)
async def list_environments(
    pipeline_path: str | None = Query(None, description="Path to pipeline file"),
) -> EnvironmentsResponse:
    """List environments available for a pipeline.

    If no pipeline_path is provided, returns the default mock environment.
    """
    environments, source = discover_environments_for_pipeline(pipeline_path, None)
    current = environments[0].name if environments else "local"

    return EnvironmentsResponse(
        environments=environments,
        current=current,
        source=source,
    )


@router.get("/{env_name}/ports")
async def get_env_ports(
    env_name: str,
    pipeline_path: str | None = Query(None, description="Path to pipeline file"),
) -> dict[str, Any]:
    """Get resolved global port configurations for an environment."""
    ports = get_environment_ports(env_name, pipeline_path, None)
    if not ports:
        raise HTTPException(status_code=404, detail=f"Environment '{env_name}' not found")
    return {"environment": env_name, "ports": ports}


@router.get("/{env_name}/node/{node_name}/port/{port_type}")
async def get_node_port(
    env_name: str,
    node_name: str,
    port_type: str,
    pipeline_path: str | None = Query(None, description="Path to pipeline file"),
    yaml_content: str | None = Query(None, description="Pipeline YAML content"),
) -> dict[str, Any]:
    """Get effective port config for a specific node.

    Returns node-level override if exists, otherwise global config.
    """
    config = get_node_port_config(env_name, node_name, port_type, pipeline_path, yaml_content)
    if not config:
        raise HTTPException(
            status_code=404,
            detail=f"No port config for node '{node_name}' "
            f"port '{port_type}' in environment '{env_name}'",
        )
    return {
        "environment": env_name,
        "node": node_name,
        "port_type": port_type,
        **config,
    }


@router.post("/save")
async def save_environment(
    config: EnvironmentConfig,
    pipeline_path: str = Query(..., description="Path to pipeline file"),
) -> dict[str, Any]:
    """Save an environment configuration to the pipeline's environments/ folder."""
    full_path = _resolve_path(pipeline_path)
    pipeline_dir = full_path.parent if full_path.is_file() else full_path

    env_dir = pipeline_dir / "environments"
    env_dir.mkdir(parents=True, exist_ok=True)

    env_file = env_dir / f"{config.name}.yaml"

    data: dict[str, Any] = {
        "description": config.description,
        "ports": {
            name: {"adapter": port.adapter, "config": port.config}
            for name, port in config.ports.items()
        },
    }

    # Include node overrides if any
    if config.node_overrides:
        data["node_overrides"] = {
            node_name: {
                port_name: {"adapter": port.adapter, "config": port.config}
                for port_name, port in node_ports.items()
            }
            for node_name, node_ports in config.node_overrides.items()
        }

    try:
        with Path(env_file).open("w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        return {"success": True, "path": str(env_file.relative_to(get_workspace_root()))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save environment: {e}") from e


@router.delete("/{env_name}")
async def delete_environment(
    env_name: str,
    pipeline_path: str = Query(..., description="Path to pipeline file"),
) -> dict[str, Any]:
    """Delete an environment configuration from the pipeline's environments/ folder."""
    full_path = _resolve_path(pipeline_path)
    pipeline_dir = full_path.parent if full_path.is_file() else full_path

    env_file = pipeline_dir / "environments" / f"{env_name}.yaml"

    if not env_file.exists():
        raise HTTPException(status_code=404, detail=f"Environment '{env_name}' not found")

    try:
        env_file.unlink()
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete environment: {e}") from e
