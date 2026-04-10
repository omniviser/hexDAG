"""Bootstrap runner -- execute the self-compile pipeline.

Provides two execution paths:

1. **YAML-first** (default) -- loads ``self_compile.yaml`` via
   :class:`~hexdag.kernel.pipeline_runner.PipelineRunner` and executes
   it through hexDAG's own orchestrator.  This is the self-compiling path:
   hexDAG building hexDAG.

2. **Programmatic fallback** -- constructs a
   :class:`~hexdag.kernel.domain.dag.DirectedGraph` directly from the
   stage functions.  Used when the YAML builder itself is broken
   (Stage-0 bootstrap).

Usage::

    # Async
    from hexdag.bootstrap.runner import run_self_compile
    results = await run_self_compile()

    # CLI
    hexdag bootstrap
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hexdag.kernel.logging import get_logger

logger = get_logger(__name__)


def self_compile_pipeline_path() -> Path:
    """Return the absolute path to ``self_compile.yaml``."""
    return Path(__file__).resolve().parent / "pipelines" / "self_compile.yaml"


async def run_self_compile(
    *,
    stages: str = "all",
    use_yaml: bool = True,
    fail_fast: bool = True,
) -> dict[str, Any]:
    """Execute the hexDAG self-compile pipeline.

    Parameters
    ----------
    stages : str
        Comma-separated stage names to run, or ``"all"`` for the full
        pipeline.  When a subset is specified, only those stages (and
        their transitive dependencies) are executed.
    use_yaml : bool
        If True (default), load the pipeline from ``self_compile.yaml``
        using hexDAG's own YAML builder (the self-compiling path).
        If False, fall back to programmatic graph construction.
    fail_fast : bool
        If True, raise on the first failed stage.

    Returns
    -------
    dict[str, Any]
        Node results keyed by stage name.

    """
    if use_yaml:
        try:
            return await _run_via_yaml(stages=stages)
        except Exception:
            logger.warning(
                "YAML pipeline execution failed -- falling back to programmatic bootstrap (Stage 0)"
            )
            return await _run_programmatic(stages=stages, fail_fast=fail_fast)
    return await _run_programmatic(stages=stages, fail_fast=fail_fast)


# ---------------------------------------------------------------------------
# YAML-first path (self-compiling)
# ---------------------------------------------------------------------------


async def _run_via_yaml(*, stages: str = "all") -> dict[str, Any]:
    """Load and execute self_compile.yaml through PipelineRunner."""
    from hexdag.kernel.pipeline_runner import PipelineRunner

    pipeline_path = self_compile_pipeline_path()
    logger.info("Self-compile: loading pipeline from {}", pipeline_path)

    runner = PipelineRunner()
    results = await runner.run(pipeline_path, input_data={})

    if stages != "all":
        requested = {s.strip() for s in stages.split(",")}
        results = {k: v for k, v in results.items() if k in requested}

    return results


# ---------------------------------------------------------------------------
# Programmatic fallback (Stage 0 bootstrap)
# ---------------------------------------------------------------------------

# Stage dependency graph (same topology as self_compile.yaml)
_STAGE_DEPS: dict[str, list[str]] = {
    "lint": [],
    "format_check": [],
    "typecheck": [],
    "validate_architecture": ["lint"],
    "validate_self": ["lint"],
    "run_tests": ["lint", "typecheck"],
    "build_package": ["run_tests", "validate_architecture", "validate_self"],
    "validate_package": ["build_package"],
}


async def _run_programmatic(
    *,
    stages: str = "all",
    fail_fast: bool = True,
) -> dict[str, Any]:
    """Build and execute the self-compile DAG programmatically.

    This is the Stage-0 bootstrap: it constructs the pipeline using
    hexDAG's Python API instead of YAML, for when the YAML builder
    itself needs to be bootstrapped.
    """
    from hexdag.bootstrap import stages as stage_mod
    from hexdag.kernel.domain.dag import DirectedGraph, NodeSpec
    from hexdag.kernel.orchestration.orchestrator import Orchestrator

    # Map stage names to async functions
    stage_fns: dict[str, Any] = {
        "lint": stage_mod.lint,
        "format_check": stage_mod.format_check,
        "typecheck": stage_mod.typecheck,
        "validate_architecture": stage_mod.validate_architecture,
        "validate_self": stage_mod.validate_self,
        "run_tests": stage_mod.run_tests,
        "build_package": stage_mod.build_package,
        "validate_package": stage_mod.validate_package,
    }

    # Determine which stages to include
    if stages == "all":
        requested = set(stage_fns.keys())
    else:
        requested = {s.strip() for s in stages.split(",")}
        # Include transitive dependencies
        requested = _resolve_deps(requested, _STAGE_DEPS)

    # Build the DAG
    nodes: list[NodeSpec] = []
    for name in stage_fns:
        if name not in requested:
            continue
        fn = stage_fns[name]
        deps = frozenset(d for d in _STAGE_DEPS[name] if d in requested)
        nodes.append(NodeSpec(name=name, fn=fn, deps=deps))

    graph = DirectedGraph(nodes=nodes)
    orchestrator = Orchestrator()

    logger.info(
        "Self-compile (programmatic): {} stages, {} waves",
        len(nodes),
        len(graph.waves()),
    )

    results = await orchestrator.run(graph, initial_input={})

    # Check for failures
    if fail_fast:
        for name, result in results.items():
            if isinstance(result, dict) and not result.get("passed", True):
                msg = f"Bootstrap stage '{name}' failed: {result.get('errors', '')}"
                raise BootstrapError(msg)

    return results


def _resolve_deps(requested: set[str], deps: dict[str, list[str]]) -> set[str]:
    """Resolve transitive dependencies for a set of requested stages."""
    resolved: set[str] = set()
    stack = list(requested)
    while stack:
        stage = stack.pop()
        if stage in resolved:
            continue
        resolved.add(stage)
        stack.extend(dep for dep in deps.get(stage, []) if dep not in resolved)
    return resolved


class BootstrapError(Exception):
    """Raised when a self-compile stage fails."""
