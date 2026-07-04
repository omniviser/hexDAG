"""The compiler front door: ``compile()``.

One entry point for every consumer — the builder, ``hexdag validate``, the
API layer, MCP tools, and Studio. Nothing outside this package may mirror
preprocessing or validation steps.

M1 status: ``compile()`` delegates to the existing ``YamlPipelineBuilder``
internals (single-sourced via ``_prepare_config``) and wraps the string-based
``ValidationReport`` into structured :class:`Diagnostic` objects with
provisional codes and no locations. Later milestones replace the delegation
with real stages (parse-with-marks, normalize, rules) without changing this
module's public surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml

from hexdag.compiler.diagnostics import (
    LEGACY_ERROR,
    LEGACY_INFO,
    LEGACY_WARNING,
    Diagnostic,
)
from hexdag.kernel.exceptions import YamlPipelineBuilderError

if TYPE_CHECKING:
    from hexdag.compiler.yaml_validator import ValidationReport
    from hexdag.kernel.domain.dag import DirectedGraph
    from hexdag.kernel.domain.pipeline_config import PipelineConfig

__all__ = ["CompileResult", "compile"]

type CompileMode = Literal["build", "validate"]


@dataclass
class CompileResult:
    """Outcome of a ``compile()`` call.

    ``graph``/``config`` are populated in build mode on success;
    ``document`` is the preprocessed pipeline document (when parseable) so
    consumers can introspect structure without re-parsing.
    """

    graph: DirectedGraph | None = None
    config: PipelineConfig | None = None
    document: dict[str, Any] | None = None
    diagnostics: list[Diagnostic] = field(default_factory=list)
    error_type: str | None = None  # exception class name for build-mode failures

    @property
    def ok(self) -> bool:
        """True when no error-severity diagnostics are present."""
        return not any(d.severity == "error" for d in self.diagnostics)

    @property
    def errors(self) -> list[str]:
        """Error messages (string view over diagnostics)."""
        return [d.message for d in self.diagnostics if d.severity == "error"]

    @property
    def warnings(self) -> list[str]:
        """Warning messages (string view over diagnostics)."""
        return [d.message for d in self.diagnostics if d.severity == "warning"]

    @property
    def suggestions(self) -> list[str]:
        """Info/suggestion messages (string view over diagnostics)."""
        return [d.message for d in self.diagnostics if d.severity == "info"]

    @property
    def node_names(self) -> list[str]:
        """Node names from the graph, or the document when not lowered."""
        if self.graph is not None:
            return [node.name for node in self.graph.values()]
        if isinstance(self.document, dict):
            return [
                n.get("metadata", {}).get("name", "unknown")
                for n in self.document.get("spec", {}).get("nodes", [])
                if isinstance(n, dict)
            ]
        return []


def compile(  # noqa: A001 - deliberate: this IS the compiler's compile()
    source: str | Path,
    *,
    mode: CompileMode = "build",
    environment: str | None = None,
    base_path: Path | None = None,
    use_cache: bool = True,
    lenient: bool = False,
    raise_on_error: bool | None = None,
) -> CompileResult:
    """Compile a pipeline manifest.

    Parameters
    ----------
    source:
        YAML content, or a path to a YAML file (``base_path`` then defaults
        to the file's parent so ``!include`` resolves like the builder).
    mode:
        ``"validate"`` runs everything up to (and including) validation and
        **never raises** — problems come back as diagnostics.
        ``"build"`` additionally lowers to a DirectedGraph; by default it
        raises :class:`YamlPipelineBuilderError` on errors (builder parity).
    lenient:
        Structure-only validation for contexts without files/env vars
        (servers, CI): ``!include`` entries are replaced by placeholders and
        preprocessing is skipped. Implies ``mode="validate"`` semantics.
    raise_on_error:
        Override the raising behavior of build mode (e.g. the API layer sets
        ``False`` to receive failures as diagnostics). Ignored in validate
        mode, which never raises.
    """
    from hexdag.compiler.yaml_builder import (  # lazy: package init imports staged first
        YamlPipelineBuilder,
    )

    yaml_content, base_path = _resolve_source(source, base_path)

    if lenient:
        return _compile_lenient(yaml_content)

    builder = YamlPipelineBuilder(base_path=base_path)

    if mode == "validate":
        return _compile_validate(builder, yaml_content, environment, use_cache)

    should_raise = True if raise_on_error is None else raise_on_error
    return _compile_build(builder, yaml_content, environment, use_cache, should_raise)


def _resolve_source(source: str | Path, base_path: Path | None) -> tuple[str, Path | None]:
    """Read Path sources; default base_path to the file's parent."""
    if isinstance(source, Path):
        if base_path is None:
            base_path = source.parent
        return source.read_text(), base_path
    return source, base_path


def _report_diagnostics(report: ValidationReport) -> list[Diagnostic]:
    """Wrap a string-based ValidationReport into structured diagnostics."""
    diags = [
        Diagnostic(code=LEGACY_ERROR, severity="error", message=m, stage="validate")
        for m in report.errors
    ]
    diags += [
        Diagnostic(code=LEGACY_WARNING, severity="warning", message=m, stage="validate")
        for m in report.warnings
    ]
    diags += [
        Diagnostic(code=LEGACY_INFO, severity="info", message=m, stage="validate")
        for m in report.suggestions
    ]
    return diags


def _error_diag(message: str, *, stage: str) -> Diagnostic:
    """Build a legacy-coded error diagnostic."""
    return Diagnostic(code=LEGACY_ERROR, severity="error", message=message, stage=stage)


def _compile_validate(
    builder: Any, yaml_content: str, environment: str | None, use_cache: bool
) -> CompileResult:
    """Stages 1-5, never raises."""
    try:
        config = builder._prepare_config(yaml_content, use_cache=use_cache, environment=environment)
    except yaml.YAMLError as e:
        return CompileResult(
            diagnostics=[_error_diag(f"YAML syntax error: {e}", stage="parse")],
            error_type="YAMLError",
        )
    except YamlPipelineBuilderError as e:
        return CompileResult(
            diagnostics=[_error_diag(str(e), stage="preprocess")],
            error_type="YamlPipelineBuilderError",
        )
    except Exception as e:  # never raise in validate mode
        return CompileResult(
            diagnostics=[_error_diag(f"{type(e).__name__}: {e}", stage="preprocess")],
            error_type=type(e).__name__,
        )

    if isinstance(config, dict) and config.get("kind") == "Macro":
        return CompileResult(document=config)

    report = builder.validator.validate(config)
    return CompileResult(
        document=config if isinstance(config, dict) else None,
        diagnostics=_report_diagnostics(report),
    )


def _compile_build(
    builder: Any,
    yaml_content: str,
    environment: str | None,
    use_cache: bool,
    should_raise: bool,
) -> CompileResult:
    """Full build. Raises like the builder unless ``raise_on_error=False``."""
    try:
        graph, config = builder.build_from_yaml_string(
            yaml_content, use_cache=use_cache, environment=environment
        )
    except Exception as e:
        if should_raise:
            raise
        return CompileResult(
            diagnostics=[_error_diag(str(e), stage="build")],
            error_type=type(e).__name__,
        )
    return CompileResult(graph=graph, config=config)


# ---------------------------------------------------------------------------
# Lenient mode (structure-only; removed in M2 when fragment mode lands)
# ---------------------------------------------------------------------------


def _strip_includes(
    obj: Any,
    warnings: list[str],
    *,
    _counter: list[int] | None = None,
) -> Any:
    """Strip ``{"!include": "..."}`` entries from parsed YAML.

    In node lists, replaces includes with placeholder data_node entries.
    Elsewhere, replaces with None. Collects a warning per stripped include.
    """
    if _counter is None:
        _counter = [0]

    if isinstance(obj, dict):
        if "!include" in obj and len(obj) == 1:
            warnings.append(f"!include '{obj['!include']}' skipped during lenient validation")
            return None
        return {k: _strip_includes(v, warnings, _counter=_counter) for k, v in obj.items()}

    if isinstance(obj, list):
        result = []
        for item in obj:
            if isinstance(item, dict) and "!include" in item and len(item) == 1:
                warnings.append(f"!include '{item['!include']}' skipped during lenient validation")
                _counter[0] += 1
                result.append({
                    "kind": "data_node",
                    "metadata": {"name": f"__included_{_counter[0]}__"},
                    "spec": {"output": {}},
                })
            else:
                result.append(_strip_includes(item, warnings, _counter=_counter))
        return result

    return obj


def _compile_lenient(yaml_content: str) -> CompileResult:
    """Structure-only validation: no files, no env vars, no templates."""
    from hexdag.compiler.yaml_validator import (  # lazy: package init imports staged first
        YamlValidator,
    )

    try:
        parsed = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        return CompileResult(
            diagnostics=[_error_diag(f"YAML syntax error: {e}", stage="parse")],
            error_type="YAMLError",
        )

    if not isinstance(parsed, dict):
        return CompileResult(
            diagnostics=[_error_diag("YAML must be a dictionary", stage="parse")],
            error_type="ParseError",
        )

    include_warnings: list[str] = []
    parsed = _strip_includes(parsed, include_warnings)

    report = YamlValidator().validate(parsed)
    diagnostics = [
        Diagnostic(code=LEGACY_WARNING, severity="warning", message=m, stage="parse")
        for m in include_warnings
    ]
    diagnostics += _report_diagnostics(report)
    return CompileResult(document=parsed, diagnostics=diagnostics)
