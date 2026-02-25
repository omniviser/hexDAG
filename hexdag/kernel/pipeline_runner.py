"""PipelineRunner — one-liner YAML pipeline execution.

Eliminates the 15-30 line boilerplate of: parse YAML → load secrets →
instantiate ports → create orchestrator → run graph.

Delegates to existing components:

- ``YamlPipelineBuilder`` — YAML → (DirectedGraph, PipelineConfig)
- ``OrchestratorFactory`` — PipelineConfig → Orchestrator (auto port instantiation)
- ``Orchestrator.run()`` — execute DAG

Examples
--------
Basic usage::

    runner = PipelineRunner()
    result = await runner.run("pipeline.yaml", input_data={"query": "hello"})

With port overrides (testing)::

    runner = PipelineRunner(port_overrides={"llm": MockLLM()})
    result = await runner.run("pipeline.yaml", input_data={"query": "hello"})

With secret provider::

    from hexdag_plugins.azure import AzureKeyVaultAdapter

    runner = PipelineRunner(
        secrets_provider=AzureKeyVaultAdapter(vault_url="https://my-vault.vault.azure.net"),
        secret_keys=["OPENAI-API-KEY", "DB-PASSWORD"],
    )
    result = await runner.run("pipeline.yaml", input_data={"query": "hello"})
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hexdag.compiler.yaml_builder import YamlPipelineBuilder
from hexdag.kernel.exceptions import PipelineRunnerError  # noqa: F401
from hexdag.kernel.logging import get_logger
from hexdag.kernel.orchestration.orchestrator_factory import OrchestratorFactory

if TYPE_CHECKING:
    from hexdag.kernel.domain.dag import DirectedGraph
    from hexdag.kernel.domain.pipeline_config import PipelineConfig
    from hexdag.kernel.orchestration.components.lifecycle_manager import (
        HookConfig,
        PostDagHookConfig,
    )
    from hexdag.kernel.ports.secret import SecretStore

logger = get_logger(__name__)

# Reuse the deferred env var pattern from component_instantiator
_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::([^}]*))?\}")


class PipelineRunner:
    """One-liner YAML pipeline execution.

    Orchestrates the full lifecycle: build → load secrets → validate env
    vars → instantiate ports → configure hooks → run.

    Parameters
    ----------
    port_overrides : dict[str, Any] | None
        Runtime port overrides. Merges with YAML-declared ports;
        overrides win on name collision.
    secrets_provider : SecretStore | None
        Secret adapter for pre-instantiation secret loading.
        Overrides any ``secret`` port declared in YAML.
    secret_keys : list[str] | None
        Specific secret keys to load. If None, loads all.
    max_concurrent_nodes : int
        Max nodes to execute concurrently (default: 10).
    strict_validation : bool
        Raise on validation failure (default: False).
    default_node_timeout : float | None
        Default per-node timeout in seconds.
    pre_hook_config : HookConfig | None
        Pre-DAG hook configuration (health checks, secrets, custom hooks).
    post_hook_config : PostDagHookConfig | None
        Post-DAG hook configuration (cleanup, checkpoints).
    base_path : Path | None
        Base path for resolving ``!include`` in YAML.
    environment : str | None
        Default environment for multi-document YAML.
    """

    def __init__(
        self,
        *,
        port_overrides: dict[str, Any] | None = None,
        secrets_provider: SecretStore | None = None,
        secret_keys: list[str] | None = None,
        max_concurrent_nodes: int = 10,
        strict_validation: bool = False,
        default_node_timeout: float | None = None,
        pre_hook_config: HookConfig | None = None,
        post_hook_config: PostDagHookConfig | None = None,
        base_path: Path | None = None,
        environment: str | None = None,
    ) -> None:
        self._port_overrides = port_overrides
        self._secrets_provider = secrets_provider
        self._secret_keys = secret_keys
        self._max_concurrent_nodes = max_concurrent_nodes
        self._strict_validation = strict_validation
        self._default_node_timeout = default_node_timeout
        self._pre_hook_config = pre_hook_config
        self._post_hook_config = post_hook_config
        self._environment = environment

        self._builder = YamlPipelineBuilder(base_path=base_path)
        self._factory = OrchestratorFactory()

        # Runner-level secret cache: skip load_to_environ on repeat runs
        self._secrets_loaded = False

    async def run(
        self,
        pipeline_path: str | Path,
        input_data: Any = None,
        *,
        environment: str | None = None,
    ) -> dict[str, Any]:
        """Run a YAML pipeline from file.

        Parameters
        ----------
        pipeline_path : str | Path
            Path to the YAML pipeline file.
        input_data : Any
            Initial input data for the pipeline.
        environment : str | None
            Environment override for multi-document YAML.

        Returns
        -------
        dict[str, Any]
            Node results keyed by node name.
        """
        path = Path(pipeline_path)
        if not path.exists():
            raise PipelineRunnerError(f"Pipeline file not found: {path}")

        yaml_content = path.read_text(encoding="utf-8")
        env = environment or self._environment

        with self._builder._temporary_base_path(path.parent):
            graph, pipeline_config = self._builder.build_from_yaml_string(
                yaml_content, environment=env
            )

        return await self._execute(graph, pipeline_config, input_data)

    async def run_from_string(
        self,
        yaml_content: str,
        input_data: Any = None,
        *,
        environment: str | None = None,
    ) -> dict[str, Any]:
        """Run a YAML pipeline from a string.

        Parameters
        ----------
        yaml_content : str
            YAML pipeline content.
        input_data : Any
            Initial input data for the pipeline.
        environment : str | None
            Environment override for multi-document YAML.

        Returns
        -------
        dict[str, Any]
            Node results keyed by node name.
        """
        env = environment or self._environment
        graph, pipeline_config = self._builder.build_from_yaml_string(yaml_content, environment=env)
        return await self._execute(graph, pipeline_config, input_data)

    async def validate(
        self,
        pipeline_path: str | Path | None = None,
        yaml_content: str | None = None,
        *,
        environment: str | None = None,
    ) -> list[str]:
        """Validate a pipeline without executing it (dry-run).

        Checks YAML parsing, DAG validity, port resolution, and env var
        availability.

        Parameters
        ----------
        pipeline_path : str | Path | None
            Path to YAML file (mutually exclusive with ``yaml_content``).
        yaml_content : str | None
            YAML string (mutually exclusive with ``pipeline_path``).
        environment : str | None
            Environment for multi-document YAML.

        Returns
        -------
        list[str]
            List of issues found. Empty list means pipeline is valid.
        """
        if pipeline_path is None and yaml_content is None:
            raise PipelineRunnerError("Provide either pipeline_path or yaml_content")

        issues: list[str] = []

        # Step 1: Parse YAML + build graph
        try:
            graph, pipeline_config = self._build(
                pipeline_path=pipeline_path,
                yaml_content=yaml_content,
                environment=environment,
            )
        except Exception as e:
            issues.append(f"YAML/DAG error: {e}")
            return issues

        # Step 2: Check env vars
        missing = _find_missing_env_vars(pipeline_config)
        issues.extend(f"Missing env var: {var}" for var in missing)

        # Step 3: Try port instantiation (without running)
        try:
            self._factory.create_orchestrator(
                pipeline_config=pipeline_config,
                additional_ports=self._port_overrides,
            )
        except Exception as e:
            issues.append(f"Port instantiation error: {e}")

        return issues

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build(
        self,
        pipeline_path: str | Path | None = None,
        yaml_content: str | None = None,
        environment: str | None = None,
    ) -> tuple[DirectedGraph, PipelineConfig]:
        """Build graph + config from file or string."""
        env = environment or self._environment

        if pipeline_path is not None:
            path = Path(pipeline_path)
            if not path.exists():
                raise PipelineRunnerError(f"Pipeline file not found: {path}")
            content = path.read_text(encoding="utf-8")
            with self._builder._temporary_base_path(path.parent):
                return self._builder.build_from_yaml_string(content, environment=env)

        if yaml_content is not None:
            return self._builder.build_from_yaml_string(yaml_content, environment=env)

        raise PipelineRunnerError("Provide either pipeline_path or yaml_content")

    async def _execute(
        self,
        graph: DirectedGraph,
        pipeline_config: PipelineConfig,
        input_data: Any,
    ) -> dict[str, Any]:
        """Core execution: secrets → validate → instantiate → run."""
        # 1. Load secrets to os.environ (before port instantiation)
        await self._load_secrets(pipeline_config)

        # 2. Validate env vars (fail-fast with all missing vars)
        missing = _find_missing_env_vars(pipeline_config)
        if missing:
            raise PipelineRunnerError(
                f"Missing environment variables required by pipeline ports: "
                f"{', '.join(missing)}. "
                f"Set them in os.environ or use a secrets_provider."
            )

        # 3. Create orchestrator (auto-instantiates ports from YAML config)
        orchestrator = self._factory.create_orchestrator(
            pipeline_config=pipeline_config,
            max_concurrent_nodes=self._max_concurrent_nodes,
            strict_validation=self._strict_validation,
            default_node_timeout=self._default_node_timeout,
            additional_ports=self._port_overrides,
            pre_hook_config=self._pre_hook_config,
            post_hook_config=self._post_hook_config,
        )

        # 4. Execute
        pipeline_name = pipeline_config.metadata.get("name", "unnamed")
        logger.info("Running pipeline '{}' with {} nodes", pipeline_name, len(graph))

        result = await orchestrator.run(graph, input_data or {})

        logger.info("Pipeline '{}' completed with {} node results", pipeline_name, len(result))
        return result

    async def _load_secrets(self, pipeline_config: PipelineConfig) -> None:
        """Load secrets into os.environ before port instantiation.

        Uses constructor ``secrets_provider`` if set, otherwise auto-detects
        a ``secret`` port from the YAML config.
        """
        if self._secrets_loaded:
            logger.debug("Secrets already loaded (cached), skipping")
            return

        provider = self._secrets_provider

        # Auto-detect secret port from YAML if no constructor override
        if provider is None and "secret" in pipeline_config.ports:
            logger.info("Auto-detected 'secret' port in YAML, loading secrets")
            try:
                adapter = self._factory.component_instantiator.instantiate_adapter(
                    pipeline_config.ports["secret"], port_name="secret"
                )
                provider = adapter
            except Exception as e:
                logger.warning("Failed to instantiate secret provider from YAML: {}", e)
                return

        if provider is None:
            return

        await provider.load_to_environ(keys=self._secret_keys)
        self._secrets_loaded = True


def _find_missing_env_vars(pipeline_config: PipelineConfig) -> list[str]:
    """Scan pipeline port configs for ``${VAR}`` patterns missing from os.environ.

    Only flags variables without a ``:default`` fallback.

    Parameters
    ----------
    pipeline_config : PipelineConfig
        The parsed pipeline configuration.

    Returns
    -------
    list[str]
        Sorted list of missing env var names.
    """
    missing: set[str] = set()

    for port_spec in pipeline_config.ports.values():
        _scan_dict_for_missing_vars(port_spec, missing)

    return sorted(missing)


def _scan_dict_for_missing_vars(obj: Any, missing: set[str]) -> None:
    """Recursively scan a dict/list/str for unresolved ``${VAR}`` references."""
    if isinstance(obj, str):
        for match in _ENV_VAR_PATTERN.finditer(obj):
            var_name = match.group(1)
            has_default = match.group(2) is not None
            if not has_default and var_name not in os.environ:
                missing.add(var_name)
    elif isinstance(obj, dict):
        for value in obj.values():
            _scan_dict_for_missing_vars(value, missing)
    elif isinstance(obj, list):
        for item in obj:
            _scan_dict_for_missing_vars(item, missing)
