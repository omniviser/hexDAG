"""Build stages for the hexDAG self-compile pipeline.

Each function is a deterministic build stage that can be referenced by
module path in a hexDAG YAML pipeline::

    - kind: function_node
      metadata:
        name: lint
      spec:
        fn: hexdag.bootstrap.stages.lint

Stage functions follow the function-node contract: they receive an
``input_data`` dict and return a result dict with at least ``stage``,
``passed``, and ``output`` keys.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from hexdag.kernel.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    """Locate the hexDAG project root (directory containing pyproject.toml).

    Walks up from this file's location.  Returns the first directory that
    contains both ``pyproject.toml`` and a ``hexdag/`` package directory.
    """
    current = Path(__file__).resolve().parent
    for ancestor in (current, *current.parents):
        if (ancestor / "pyproject.toml").exists() and (ancestor / "hexdag").is_dir():
            return ancestor
    msg = "Could not locate hexDAG project root"
    raise RuntimeError(msg)


async def _run_command(
    *args: str,
    cwd: Path | None = None,
    timeout: float = 300.0,
) -> tuple[int, str, str]:
    """Run a subprocess and return (returncode, stdout, stderr).

    Parameters
    ----------
    *args : str
        Command and arguments.
    cwd : Path | None
        Working directory (defaults to project root).
    timeout : float
        Maximum execution time in seconds.
    """
    cwd = cwd or _project_root()
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.communicate()
        return 1, "", f"Command timed out after {timeout}s: {' '.join(args)}"

    return (
        proc.returncode or 0,
        (stdout_bytes or b"").decode(errors="replace"),
        (stderr_bytes or b"").decode(errors="replace"),
    )


def _stage_result(
    stage: str,
    passed: bool,
    output: str,
    errors: str = "",
    **extra: Any,
) -> dict[str, Any]:
    """Build a standardised stage result dict."""
    result: dict[str, Any] = {
        "stage": stage,
        "passed": passed,
        "output": output,
    }
    if errors:
        result["errors"] = errors
    result.update(extra)
    return result


# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------


async def lint(input_data: dict[str, Any]) -> dict[str, Any]:
    """Run the ruff linter on hexdag source code.

    Parameters
    ----------
    input_data : dict
        Pipeline input (unused by root stages).

    Returns
    -------
    dict
        Stage result with ``passed``, ``output``, ``errors`` keys.
    """
    rc, stdout, stderr = await _run_command("uv", "run", "ruff", "check", "hexdag/")
    return _stage_result("lint", passed=rc == 0, output=stdout, errors=stderr)


async def format_check(input_data: dict[str, Any]) -> dict[str, Any]:
    """Verify source code formatting with ruff format --check.

    Parameters
    ----------
    input_data : dict
        Pipeline input (unused by root stages).

    Returns
    -------
    dict
        Stage result.
    """
    rc, stdout, stderr = await _run_command("uv", "run", "ruff", "format", "--check", "hexdag/")
    return _stage_result("format_check", passed=rc == 0, output=stdout, errors=stderr)


async def typecheck(input_data: dict[str, Any]) -> dict[str, Any]:
    """Run pyright type checker on hexdag source code.

    Parameters
    ----------
    input_data : dict
        Pipeline input (unused by root stages).

    Returns
    -------
    dict
        Stage result.
    """
    rc, stdout, stderr = await _run_command("uv", "run", "pyright", "hexdag/")
    return _stage_result("typecheck", passed=rc == 0, output=stdout, errors=stderr)


async def validate_architecture(input_data: dict[str, Any]) -> dict[str, Any]:
    """Run hexDAG architecture validation scripts.

    Executes the suite of ``scripts/check_*.py`` validators that enforce
    hexagonal architecture, kernel boundaries, async I/O rules, and more.

    Parameters
    ----------
    input_data : dict
        Receives output from predecessor stages (e.g. lint).

    Returns
    -------
    dict
        Stage result with per-check breakdown.
    """
    root = _project_root()
    checks = [
        "check_core_imports",
        "check_kernel_boundary",
        "check_async_io",
        "check_exception_hierarchy",
        "check_port_protocols",
        "check_circular_imports",
    ]

    results: list[dict[str, Any]] = []
    all_passed = True

    for check in checks:
        script = root / "scripts" / f"{check}.py"
        if not script.exists():
            results.append({"check": check, "passed": True, "skipped": True})
            continue

        rc, stdout, stderr = await _run_command("uv", "run", "python", str(script))
        passed = rc == 0
        if not passed:
            all_passed = False
        results.append({
            "check": check,
            "passed": passed,
            "output": stdout[:500],
            "errors": stderr[:500],
        })

    summary_lines = [
        f"  {'PASS' if r['passed'] else 'FAIL'} {r['check']}"
        + (" (skipped)" if r.get("skipped") else "")
        for r in results
    ]
    summary = "\n".join(summary_lines)

    return _stage_result(
        "validate_architecture",
        passed=all_passed,
        output=summary,
        checks=results,
    )


async def validate_self(input_data: dict[str, Any]) -> dict[str, Any]:
    """Validate the self-compile pipeline YAML using hexDAG's own validator.

    This is the meta-recursive proof of self-consistency: hexDAG validates
    the very pipeline that builds hexDAG.

    Parameters
    ----------
    input_data : dict
        Receives output from predecessor stages.

    Returns
    -------
    dict
        Stage result.
    """
    from hexdag.bootstrap.runner import self_compile_pipeline_path

    pipeline_path = self_compile_pipeline_path()

    if not pipeline_path.exists():
        return _stage_result(
            "validate_self",
            passed=False,
            output="",
            errors=f"Self-compile pipeline not found: {pipeline_path}",
        )

    # Use hexDAG's own YAML pipeline builder to parse and validate
    try:
        from hexdag.compiler.yaml_builder import YamlPipelineBuilder

        builder = YamlPipelineBuilder()
        graph, _pipeline_config = builder.build_from_yaml_file(str(pipeline_path))

        node_count = len(graph.nodes)
        wave_count = len(graph.waves())

        output = (
            f"Self-compile pipeline validated successfully.\n"
            f"  Nodes: {node_count}\n"
            f"  Waves: {wave_count}\n"
            f"  Pipeline can compile itself."
        )
        return _stage_result("validate_self", passed=True, output=output)

    except Exception as exc:
        return _stage_result(
            "validate_self",
            passed=False,
            output="",
            errors=f"Self-validation failed: {exc}",
        )


async def run_tests(input_data: dict[str, Any]) -> dict[str, Any]:
    """Run the hexDAG test suite via pytest.

    Parameters
    ----------
    input_data : dict
        Receives output from predecessor stages (lint, typecheck).

    Returns
    -------
    dict
        Stage result with test counts.
    """
    rc, stdout, stderr = await _run_command(
        "uv",
        "run",
        "pytest",
        "tests/hexdag/",
        "-x",
        "--tb=short",
        "-q",
        timeout=600.0,
    )
    return _stage_result("run_tests", passed=rc == 0, output=stdout, errors=stderr)


async def build_package(input_data: dict[str, Any]) -> dict[str, Any]:
    """Build the hexDAG Python package (wheel + sdist).

    Parameters
    ----------
    input_data : dict
        Receives output from predecessor stages.

    Returns
    -------
    dict
        Stage result with built artifact paths.
    """
    root = _project_root()

    # Clean previous build artifacts
    dist_dir = root / "dist"
    if dist_dir.exists():
        import shutil

        shutil.rmtree(dist_dir)

    rc, stdout, stderr = await _run_command(
        "uv",
        "run",
        "python",
        "-m",
        "build",
        timeout=120.0,
    )

    artifacts: list[str] = []
    if rc == 0 and dist_dir.exists():
        artifacts = [p.name for p in dist_dir.iterdir()]

    return _stage_result(
        "build_package",
        passed=rc == 0,
        output=stdout,
        errors=stderr,
        artifacts=artifacts,
    )


async def validate_package(input_data: dict[str, Any]) -> dict[str, Any]:
    """Validate the built package artifacts.

    Checks that wheel and sdist were produced and can be inspected.

    Parameters
    ----------
    input_data : dict
        Receives output from ``build_package`` stage.

    Returns
    -------
    dict
        Stage result.
    """
    root = _project_root()
    dist_dir = root / "dist"

    if not dist_dir.exists():
        return _stage_result(
            "validate_package",
            passed=False,
            output="",
            errors="dist/ directory not found -- was build_package run?",
        )

    artifacts = list(dist_dir.iterdir())
    wheels = [a for a in artifacts if a.suffix == ".whl"]
    sdists = [a for a in artifacts if a.name.endswith(".tar.gz")]

    issues: list[str] = []
    if not wheels:
        issues.append("No wheel (.whl) found in dist/")
    if not sdists:
        issues.append("No sdist (.tar.gz) found in dist/")

    # Validate with twine if available
    rc, stdout, stderr = await _run_command(
        "uv",
        "run",
        "twine",
        "check",
        "dist/*",
        timeout=30.0,
    )
    if rc != 0:
        issues.append(f"twine check failed: {stderr[:300]}")

    passed = len(issues) == 0
    output_lines = [f"  {a.name}" for a in artifacts]
    output = "Built artifacts:\n" + "\n".join(output_lines)

    return _stage_result(
        "validate_package",
        passed=passed,
        output=output,
        errors="\n".join(issues) if issues else "",
        artifact_count=len(artifacts),
    )
