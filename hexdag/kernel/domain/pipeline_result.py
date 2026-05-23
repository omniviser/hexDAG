"""PipelineResult — structured return value from pipeline execution.

When a pipeline declares ``spec.output``, the runner maps node results
to named output fields so callers don't need to know pipeline internals.

Example YAML::

    spec:
      output:
        action: router.action
        email_body: router.response_text
        confidence: extraction.confidence_score

Example usage::

    result = await runner.run("pipeline.yaml", {"load_id": 123})
    result.output["action"]       # "counter"
    result.node_results["router"] # full node output dict
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import ItemsView, Iterator, KeysView, ValuesView


@dataclass(slots=True)
class PipelineResult:
    """Structured result from a pipeline execution.

    Backwards-compatible: supports ``result["node_name"]`` dict-style
    access via ``__getitem__`` so existing code keeps working.
    New code should use ``result.output`` for declared outputs.

    Attributes
    ----------
    node_results : dict[str, Any]
        Full node results keyed by node name (same as the raw dict
        returned by ``Orchestrator.run()``).
    output : dict[str, Any]
        Declared output fields mapped from node results via
        ``spec.output``.  Empty dict if the pipeline has no output
        declaration.
    pipeline_name : str
        Name of the executed pipeline.
    """

    node_results: dict[str, Any]
    output: dict[str, Any] = field(default_factory=dict)
    pipeline_name: str = ""
    status: str = "completed"
    run_id: str = ""
    suspend_metadata: dict[str, Any] | None = None

    # Backwards-compatible dict-style access
    def __getitem__(self, key: str) -> Any:
        return self.node_results[key]

    def __contains__(self, key: object) -> bool:
        return key in self.node_results

    def __iter__(self) -> Iterator[str]:
        return iter(self.node_results)

    def __len__(self) -> int:
        return len(self.node_results)

    def keys(self) -> KeysView[str]:
        """Return node result keys (backwards-compatible)."""
        return self.node_results.keys()

    def values(self) -> ValuesView[Any]:
        """Return node result values (backwards-compatible)."""
        return self.node_results.values()

    def items(self) -> ItemsView[str, Any]:
        """Return node result items (backwards-compatible)."""
        return self.node_results.items()

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-style get for backwards compatibility."""
        return self.node_results.get(key, default)


def resolve_output(
    node_results: dict[str, Any],
    output_mapping: dict[str, str],
) -> dict[str, Any]:
    """Resolve declared output fields from node results.

    Parameters
    ----------
    node_results : dict[str, Any]
        Full node results keyed by node name.
    output_mapping : dict[str, str]
        Mapping of ``{output_field: "node_name.field.subfield"}``.

    Returns
    -------
    dict[str, Any]
        Resolved output values.  Missing paths resolve to ``None``.

    Examples
    --------
    >>> results = {"router": {"action": "counter", "text": "hi"}}
    >>> resolve_output(results, {"action": "router.action"})
    {'action': 'counter'}
    """
    output: dict[str, Any] = {}
    for alias, path in output_mapping.items():
        parts = path.split(".")
        value: Any = node_results
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                value = None
                break
        output[alias] = value
    return output
