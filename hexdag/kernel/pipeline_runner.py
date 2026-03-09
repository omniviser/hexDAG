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
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hexdag.compiler.yaml_builder import YamlPipelineBuilder
from hexdag.kernel.exceptions import PipelineRunnerError  # noqa: F401
from hexdag.kernel.logging import configure_logging, get_logger
from hexdag.kernel.orchestration.orchestrator_factory import OrchestratorFactory
from hexdag.kernel.utils.node_timer import Timer

if TYPE_CHECKING:
    from hexdag.kernel.config.models import HexDAGConfig
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

    Configuration follows a four-level override chain::

        kind: Config defaults → constructor args → kind: Pipeline spec → runtime

    When ``config`` is provided, its ``orchestrator`` settings act as defaults.
    Explicit constructor args (``max_concurrent_nodes`` etc.) override the config.
    Per-pipeline ``spec.orchestrator`` in the Pipeline YAML overrides both.

    Parameters
    ----------
    config : HexDAGConfig | None
        Organisation-wide defaults loaded from ``kind: Config``.
        Used as fallback when explicit params are not set.
    port_overrides : dict[str, Any] | None
        Runtime port overrides. Merges with YAML-declared ports;
        overrides win on name collision.
    secrets_provider : SecretStore | None
        Secret adapter for pre-instantiation secret loading.
        Overrides any ``secret`` port declared in YAML.
    secret_keys : list[str] | None
        Specific secret keys to load. If None, loads all.
    max_concurrent_nodes : int | None
        Max nodes to execute concurrently. None = use config or default (10).
    strict_validation : bool | None
        Raise on validation failure. None = use config or default (False).
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
        config: HexDAGConfig | None = None,
        port_overrides: dict[str, Any] | None = None,
        secrets_provider: SecretStore | None = None,
        secret_keys: list[str] | None = None,
        max_concurrent_nodes: int | None = None,
        strict_validation: bool | None = None,
        default_node_timeout: float | None = None,
        pre_hook_config: HookConfig | None = None,
        post_hook_config: PostDagHookConfig | None = None,
        base_path: Path | None = None,
        environment: str | None = None,
    ) -> None:
        self._config = config
        self._port_overrides = port_overrides
        self._secrets_provider = secrets_provider
        self._secret_keys = secret_keys

        # Store raw constructor args (None = not set by caller).
        # These are kept separate from config/hardcoded defaults so that
        # _resolve_orchestrator_settings() can implement the full priority
        # chain: explicit constructor > per-pipeline spec > config > hardcoded.
        self._explicit_max_concurrent_nodes = max_concurrent_nodes
        self._explicit_strict_validation = strict_validation
        self._explicit_default_node_timeout = default_node_timeout

        # Resolve constructor-level defaults: explicit arg > config > hardcoded.
        # These are the "baseline" values used when no per-pipeline spec exists.
        cfg_orch = config.orchestrator if config else None
        self._max_concurrent_nodes = (
            max_concurrent_nodes
            if max_concurrent_nodes is not None
            else (cfg_orch.max_concurrent_nodes if cfg_orch else 10)
        )
        self._strict_validation = (
            strict_validation
            if strict_validation is not None
            else (cfg_orch.strict_validation if cfg_orch else False)
        )
        self._default_node_timeout = (
            default_node_timeout
            if default_node_timeout is not None
            else (cfg_orch.default_node_timeout if cfg_orch else None)
        )

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

        effective_config = self._effective_config()
        return await self._execute(graph, pipeline_config, input_data, effective_config)

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
        effective_config = self._effective_config()
        return await self._execute(graph, pipeline_config, input_data, effective_config)

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

    def _effective_config(self) -> Any:
        """Return the effective HexDAGConfig for the current build.

        Merges the inline ``kind: Config`` from multi-doc YAML with the
        constructor-provided config.  Constructor config wins on conflict.
        If neither exists, returns ``None``.
        """
        inline = self._builder.inline_config
        if inline is None:
            return self._config
        if self._config is None:
            return inline
        # Both exist — constructor config takes precedence.
        # Return the constructor config (it's the explicitly-provided one).
        return self._config

    async def _execute(
        self,
        graph: DirectedGraph,
        pipeline_config: PipelineConfig,
        input_data: Any,
        effective_config: Any = None,
    ) -> dict[str, Any]:
        """Core execution: secrets → validate → instantiate → run.

        Parameters
        ----------
        graph:
            Compiled DAG.
        pipeline_config:
            Parsed pipeline configuration.
        input_data:
            Initial input for root nodes.
        effective_config:
            The effective ``HexDAGConfig`` for this run (may include
            inline ``kind: Config`` from the YAML file).
        """
        # 0. Apply logging configuration from kind: Config (if present)
        if effective_config and effective_config.logging:
            log_cfg = effective_config.logging
            configure_logging(
                level=log_cfg.level,
                format=log_cfg.format,
                output_file=log_cfg.output_file,
                use_color=log_cfg.use_color,
                include_timestamp=log_cfg.include_timestamp,
                use_rich=log_cfg.use_rich,
                dual_sink=log_cfg.dual_sink,
                enable_stdlib_bridge=log_cfg.enable_stdlib_bridge,
                backtrace=log_cfg.backtrace,
                diagnose=log_cfg.diagnose,
            )

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

        # 3. Resolve final orchestrator settings.
        #    Priority: explicit constructor arg > per-pipeline spec > config > hardcoded
        #    Use effective_config (which may come from inline kind: Config)
        #    for the config-level defaults when no constructor config was given.
        max_nodes, strict, timeout = self._resolve_orchestrator_settings(
            pipeline_config.orchestrator,
            effective_config,
        )

        # 4. Resolve effective limits and caps.
        #    Priority: per-pipeline spec > effective config > None.
        #    Enforcement is not yet implemented (see Steps 2-3 in ROADMAP);
        #    this ensures the values are resolved and logged so they're
        #    ready for the ResourceAccounting / CapSet hooks.
        effective_limits = self._resolve_field(
            pipeline_config.limits,
            effective_config.limits if effective_config else None,
        )
        effective_caps = self._resolve_field(
            pipeline_config.caps,
            effective_config.caps if effective_config else None,
        )
        if effective_limits is not None:
            logger.debug(
                "Resource limits active: {}",
                {k: v for k, v in asdict(effective_limits).items() if v is not None},
            )
        if effective_caps is not None:
            logger.debug("Capability caps active: {}", asdict(effective_caps))

        # 5. Create orchestrator (auto-instantiates ports from YAML config)
        orchestrator = self._factory.create_orchestrator(
            pipeline_config=pipeline_config,
            max_concurrent_nodes=max_nodes,
            strict_validation=strict,
            default_node_timeout=timeout,
            additional_ports=self._port_overrides,
            pre_hook_config=self._pre_hook_config,
            post_hook_config=self._post_hook_config,
        )

        # 6. Execute
        pipeline_name = pipeline_config.metadata.get("name", "unnamed")
        logger.info("Running pipeline '{}' with {} nodes", pipeline_name, len(graph))

        t = Timer()
        result = await orchestrator.run(graph, input_data or {})

        logger.info(
            "Pipeline '{}' completed with {} node results in {}",
            pipeline_name,
            len(result),
            t.duration_str,
        )
        return result

    def _resolve_orchestrator_settings(
        self,
        pc_orch: Any,
        effective_config: Any = None,
    ) -> tuple[int, bool, float | None]:
        """Resolve final orchestrator settings with correct priority.

        Priority chain (highest to lowest):
        1. Explicit constructor args (``max_concurrent_nodes=20``)
        2. Per-pipeline spec (``pipeline_config.orchestrator``)
        3. Effective config defaults (constructor config or inline ``kind: Config``)
        4. Hardcoded defaults (10, False, None)

        Parameters
        ----------
        pc_orch:
            ``pipeline_config.orchestrator`` — may be ``None``.
        effective_config:
            The effective ``HexDAGConfig`` for this run.  May differ from
            ``self._config`` when the YAML file contains an inline
            ``kind: Config`` document.

        Returns
        -------
        tuple[int, bool, float | None]
            (max_concurrent_nodes, strict_validation, default_node_timeout)
        """
        # Start from effective config (which may include inline kind: Config)
        # or fall back to constructor-resolved baseline.
        run_config = effective_config or self._config
        cfg_orch = run_config.orchestrator if run_config else None

        # Level 3+4: config > hardcoded
        max_nodes = cfg_orch.max_concurrent_nodes if cfg_orch else 10
        strict = cfg_orch.strict_validation if cfg_orch else False
        timeout = cfg_orch.default_node_timeout if cfg_orch else None

        # Level 2: per-pipeline spec overrides config (but only for
        # fields not explicitly set by the constructor).
        if pc_orch is not None:
            if self._explicit_max_concurrent_nodes is None:
                max_nodes = pc_orch.max_concurrent_nodes
            if self._explicit_strict_validation is None:
                strict = pc_orch.strict_validation
            if (
                self._explicit_default_node_timeout is None
                and pc_orch.default_node_timeout is not None
            ):
                timeout = pc_orch.default_node_timeout

        # Level 1: explicit constructor args always win.
        if self._explicit_max_concurrent_nodes is not None:
            max_nodes = self._explicit_max_concurrent_nodes
        if self._explicit_strict_validation is not None:
            strict = self._explicit_strict_validation
        if self._explicit_default_node_timeout is not None:
            timeout = self._explicit_default_node_timeout

        return max_nodes, strict, timeout

    @staticmethod
    def _resolve_field(pipeline_value: Any, config_value: Any) -> Any:
        """Return pipeline-level override if set, else config-level default."""
        if pipeline_value is not None:
            return pipeline_value
        return config_value

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
