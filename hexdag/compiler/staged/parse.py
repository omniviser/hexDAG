"""Parse stage: YAML → documents + source map, with ``!include`` expansion.

Values come from ``yaml.safe_load_all`` (custom tags like ``!py`` behave
exactly as before); provenance comes from a parallel ``yaml.compose_all``
node tree. ``!include`` directives are expanded *here*, while marks are
alive, so every element remembers the file and line it came from — even
after fragments are spliced into the including document.

Security and semantics ported from the retired include preprocessing
plugin: relative paths only, no traversal outside the project root,
circular-include detection, max nesting depth, ``file.yaml#anchor``
sub-document extraction, and list splicing for fragment includes.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from hexdag.compiler.diagnostics import Location
from hexdag.compiler.source_map import SourceMap, SourcePath
from hexdag.kernel.exceptions import YamlPipelineBuilderError

INCLUDE_KEY = "!include"
_MAX_DEPTH = 10

__all__ = ["IncludeError", "ParsedSource", "parse_source"]


class IncludeError(YamlPipelineBuilderError):
    """An ``!include`` directive failed; carries the directive's location."""

    def __init__(self, message: str, loc: Location | None = None) -> None:
        """Store *loc* (the ``!include`` site in the including file)."""
        super().__init__(message)
        self.loc = loc


@dataclass(frozen=True)
class ParsedSource:
    """Parsed documents (includes expanded) plus their source map."""

    docs: tuple[Any, ...]
    source_map: SourceMap
    entry_file: str | None = None


@dataclass
class _ParseContext:
    """Shared state for one parse run."""

    project_root: Path
    source_map: SourceMap
    include_stack: list[Path] = field(default_factory=list)
    max_depth: int = _MAX_DEPTH


def parse_source(
    yaml_content: str,
    *,
    entry_file: str | None = None,
    base_path: Path | None = None,
    project_root: Path | None = None,
    max_depth: int = _MAX_DEPTH,
) -> ParsedSource:
    """Parse YAML content into documents with include expansion + provenance.

    Parameters
    ----------
    yaml_content:
        The YAML source (single- or multi-document).
    entry_file:
        Display name for the entry source (used in locations).
    base_path:
        Directory for resolving relative ``!include`` paths. Defaults to CWD.
    project_root:
        Security jail: includes may not resolve outside this directory.
        Defaults to *base_path* (matching the retired include plugin, where
        the root was fixed at construction while the base moved per file).
    """
    base = base_path or Path.cwd()
    ctx = _ParseContext(
        project_root=project_root or base, source_map=SourceMap(), max_depth=max_depth
    )

    pairs = _load_with_nodes(yaml_content)
    docs = []
    for i, (data, node) in enumerate(pairs):
        # Single-document sources use bare paths; multi-doc prefixes the index
        prefix: SourcePath = (i,) if len(pairs) > 1 else ()
        docs.append(_build(data, node, entry_file, prefix, base, 0, ctx))
    return ParsedSource(docs=tuple(docs), source_map=ctx.source_map, entry_file=entry_file)


def _load_with_nodes(content: str) -> list[tuple[Any, yaml.Node | None]]:
    """Load values (constructors run) and compose nodes (marks) in parallel."""
    try:
        datas = list(yaml.safe_load_all(content))
        nodes: list[yaml.Node | None] = list(yaml.compose_all(content))
    except yaml.YAMLError:
        raise
    if len(nodes) != len(datas):  # defensive; should not happen
        nodes = [None] * len(datas)
    return list(zip(datas, nodes, strict=True))


def _loc(node: yaml.Node | None, file: str | None, path: SourcePath) -> Location | None:
    if node is None or node.start_mark is None:
        return None
    return Location(
        file=file, line=node.start_mark.line + 1, column=node.start_mark.column + 1, path=path
    )


def _is_include(value: Any) -> bool:
    return isinstance(value, dict) and len(value) == 1 and INCLUDE_KEY in value


def _build(
    data: Any,
    node: yaml.Node | None,
    file: str | None,
    path: SourcePath,
    base: Path,
    depth: int,
    ctx: _ParseContext,
) -> Any:
    """Recursively construct the final value, recording marks per path."""
    # Resolve include chains at this position (an included file may itself
    # be a single !include). base/file/depth travel with the resolution.
    # Resolved files stay on the include stack while their content is
    # built, so nested includes see their ancestors (circular detection).
    pushed = 0
    while _is_include(data):
        data, node, file, base, depth = _load_include(data, node, file, path, base, depth, ctx)
        pushed += 1

    try:
        if loc := _loc(node, file, path):
            ctx.source_map.record(path, loc)

        if isinstance(data, dict):
            child_nodes = _mapping_child_nodes(node)
            return {
                key: _build(value, child_nodes.get(key), file, (*path, key), base, depth, ctx)
                for key, value in data.items()
            }

        if isinstance(data, list):
            item_nodes = _sequence_child_nodes(node, len(data))
            result: list[Any] = []
            for item, item_node in zip(data, item_nodes, strict=True):
                if _is_include(item):
                    value, v_node, v_file, v_base, v_depth = _load_include(
                        item, item_node, file, (*path, len(result)), base, depth, ctx
                    )
                    try:
                        if isinstance(value, list):
                            # Fragment splice: extend, keeping element origins
                            el_nodes = _sequence_child_nodes(v_node, len(value))
                            for el, el_node in zip(value, el_nodes, strict=True):
                                result.append(
                                    _build(
                                        el,
                                        el_node,
                                        v_file,
                                        (*path, len(result)),
                                        v_base,
                                        v_depth,
                                        ctx,
                                    )
                                )
                        else:
                            idx = len(result)
                            result.append(
                                _build(value, v_node, v_file, (*path, idx), v_base, v_depth, ctx)
                            )
                    finally:
                        ctx.include_stack.pop()
                    continue
                result.append(_build(item, item_node, file, (*path, len(result)), base, depth, ctx))
            return result

        return data
    finally:
        for _ in range(pushed):
            ctx.include_stack.pop()


def _mapping_child_nodes(node: yaml.Node | None) -> dict[Any, yaml.Node]:
    """Key → value-node map for a MappingNode (best-effort; merge keys skip)."""
    if not isinstance(node, yaml.MappingNode):
        return {}
    children: dict[Any, yaml.Node] = {}
    for key_node, value_node in node.value:
        if isinstance(key_node, yaml.ScalarNode):
            children[key_node.value] = value_node
    return children


def _sequence_child_nodes(node: yaml.Node | None, count: int) -> list[yaml.Node | None]:
    """Item nodes for a SequenceNode, or Nones when unavailable/misaligned."""
    if isinstance(node, yaml.SequenceNode) and len(node.value) == count:
        return list(node.value)
    return [None] * count


def _load_include(
    data: dict[str, Any],
    node: yaml.Node | None,
    file: str | None,
    path: SourcePath,
    base: Path,
    depth: int,
    ctx: _ParseContext,
) -> tuple[Any, yaml.Node | None, str | None, Path, int]:
    """Resolve one ``!include`` directive to (value, node, file, base, depth).

    On success the included file is left ON the include stack — the caller
    pops it after the included content is fully built, so nested includes
    can detect circular ancestry.
    """
    site = _loc(node, file, path)

    if depth >= ctx.max_depth:
        chain = " -> ".join(str(p) for p in ctx.include_stack)
        raise IncludeError(
            f"Include nesting too deep (max {ctx.max_depth}). "
            f"Possible circular include in: {chain}",
            site,
        )

    include_spec = str(data[INCLUDE_KEY]).strip()
    if "#" in include_spec:
        file_path_str, anchor = (part.strip() for part in include_spec.split("#", 1))
    else:
        file_path_str, anchor = include_spec, None

    file_path = _resolve_include_path(file_path_str, base, ctx, site)

    if file_path in ctx.include_stack:
        cycle = " -> ".join(str(p) for p in [*ctx.include_stack, file_path])
        raise IncludeError(f"Circular include detected: {cycle}", site)

    try:
        content = file_path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise IncludeError(
            f"Include file not found: {file_path}\nSearched relative to: {base}", site
        ) from e

    ctx.include_stack.append(file_path)
    try:
        try:
            pairs = _load_with_nodes(content)
        except yaml.YAMLError as e:
            raise IncludeError(f"Invalid YAML in included file {file_path}: {e}", site) from e
        value, value_node = pairs[0] if pairs else (None, None)

        if anchor:
            if not isinstance(value, dict) or anchor not in value:
                available = list(value.keys()) if isinstance(value, dict) else "N/A"
                raise IncludeError(
                    f"Anchor '{anchor}' not found in {file_path}. Available: {available}", site
                )
            anchor_node = _mapping_child_nodes(value_node).get(anchor)
            value, value_node = value[anchor], anchor_node
    except BaseException:
        ctx.include_stack.pop()
        raise

    # Success: file stays on the stack until its content is built (caller pops)
    return value, value_node, str(file_path), file_path.parent, depth + 1


def _resolve_include_path(
    path_str: str, base: Path, ctx: _ParseContext, site: Location | None
) -> Path:
    """Validate and resolve an include path (relative-only, root-jailed)."""
    if Path(path_str).is_absolute():
        raise IncludeError(
            f"Absolute paths not allowed in !include: {path_str}\n"
            "Use relative paths only for security.",
            site,
        )

    resolved = (base / path_str).resolve()
    resolved_root = ctx.project_root.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as e:
        raise IncludeError(
            f"Include path traverses outside project root: {path_str}\n"
            f"Project root: {resolved_root}\n"
            f"Attempted path: {resolved}",
            site,
        ) from e
    return resolved


# ---------------------------------------------------------------------------
# Parse cache: only for sources without includes (file contents can change
# independently of the entry source, so include-bearing sources never cache).
# ---------------------------------------------------------------------------

_CACHE: dict[str, ParsedSource] = {}
_CACHE_LIMIT = 32


def parse_source_cached(
    yaml_content: str,
    *,
    entry_file: str | None = None,
    base_path: Path | None = None,
    project_root: Path | None = None,
) -> ParsedSource:
    """Like :func:`parse_source`, with caching for include-free sources.

    Cached documents are deep-copied on return so downstream mutation can
    never poison the cache. Sources containing ``!include`` are never cached
    (included files can change independently of the entry source; the paths
    also depend on base_path/project_root).
    """
    if INCLUDE_KEY in yaml_content:
        return parse_source(
            yaml_content, entry_file=entry_file, base_path=base_path, project_root=project_root
        )

    cached = _CACHE.get(yaml_content)
    if cached is None:
        cached = parse_source(yaml_content, entry_file=entry_file, base_path=base_path)
        if len(_CACHE) >= _CACHE_LIMIT:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[yaml_content] = cached
    return ParsedSource(
        docs=copy.deepcopy(cached.docs),
        source_map=cached.source_map,
        entry_file=cached.entry_file,
    )
