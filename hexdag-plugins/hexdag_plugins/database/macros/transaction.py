"""TransactionMacro — savepoint-scoped partial rollback for a group of steps.

Brackets a set of child nodes in a SQLAlchemy SAVEPOINT so the group is
atomic *within* the pipeline run.  On success the savepoint is released (its
writes fold into the run transaction); if any wrapped step fails, only that
scope is rolled back while the rest of the run still commits.

YAML usage::

    - kind: macro_invocation
      metadata: { name: resolve_writes }
      spec:
        macro: transaction
        config:
          service: database        # name of the DatabaseService in spec.services
          nodes:                    # the steps to run inside the savepoint
            - kind: step_call
              metadata: { name: resolve_record }
              spec: { service: database, method: resolve_escalation_record, ... }
            - kind: step_call
              metadata: { name: release_hold }
              spec: { service: database, method: release_load_hold, ... }
        dependencies: [get_context]

Expansion shape (scope ``S`` = the macro instance name)::

    S__tx_begin  -> child_1 ... child_n -> S__tx_agg -> S__tx_release  (when: not failed)
                                                     -> S__tx_rollback (when: failed)

Each child gets ``on_error: S__tx_agg`` so a step failure is tolerated (the run
continues) and surfaces as an ``_error`` payload that the aggregator detects.

.. note::
    Only **one** ``transaction`` scope is supported per pipeline run.  Two scopes
    in a single run share one session and their SAVEPOINTs overlap (rolling one
    back discards the other), so a second scope is rejected at runtime.  For
    multiple independent atomic units, model each as its own process in a
    ``kind: System`` — one process is one run is one transaction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hexdag.kernel.configurable import ConfigurableMacro, MacroConfig
from hexdag.kernel.exceptions import YamlPipelineBuilderError

if TYPE_CHECKING:
    from collections.abc import Callable

    from hexdag.kernel.domain.dag import DirectedGraph


class TransactionMacroConfig(MacroConfig):
    """Configuration for :class:`TransactionMacro`."""

    service: str = "database"
    nodes: list[dict[str, Any]] = []  # child node specs to wrap in the savepoint


class TransactionMacro(ConfigurableMacro, yaml_alias="transaction"):
    """Wrap child steps in a SAVEPOINT scope with partial rollback on failure."""

    Config = TransactionMacroConfig

    def expand(
        self,
        instance_name: str,
        inputs: dict[str, Any],
        dependencies: list[str],
        node_builder: Callable[[list[dict[str, Any]]], DirectedGraph] | None = None,
    ) -> DirectedGraph:
        if node_builder is None:  # pragma: no cover - compiler always injects it
            raise YamlPipelineBuilderError(
                "TransactionMacro requires a node_builder (build via the compiler)"
            )

        config: TransactionMacroConfig = self.config  # type: ignore[assignment]
        service = config.service
        # Accept child nodes from config.nodes (preferred) or inputs.nodes.
        child_specs: list[dict[str, Any]] = list(config.nodes or inputs.get("nodes") or [])
        if not child_specs:
            raise YamlPipelineBuilderError(
                f"transaction macro '{instance_name}' has no child nodes (set spec.config.nodes)"
            )

        scope = instance_name
        begin_name = f"{scope}__tx_begin"
        agg_name = f"{scope}__tx_agg"
        release_name = f"{scope}__tx_release"
        rollback_name = f"{scope}__tx_rollback"
        scope_literal = f"'{scope}'"  # constant string for input_mapping

        def _savepoint_node(node_name: str, method: str, when: str | None) -> dict[str, Any]:
            spec: dict[str, Any] = {
                "service": service,
                "method": method,
                "input_mapping": {"scope_id": scope_literal},
            }
            if when is not None:
                spec["when"] = when
            return {
                "kind": "service_call_node",
                "metadata": {"name": node_name},
                "spec": spec,
            }

        # 1. Begin SAVEPOINT (sole entry node — external deps attach here).
        begin = _savepoint_node(begin_name, "begin_savepoint", when=None)
        begin["dependencies"] = []

        # 2. Children: run after begin, tolerate failure (route to aggregator).
        child_names: list[str] = []
        wrapped_children: list[dict[str, Any]] = []
        for child in child_specs:
            child = {**child}
            name = child.get("metadata", {}).get("name")
            if not name:
                raise YamlPipelineBuilderError(
                    f"transaction macro '{instance_name}': every child node needs metadata.name"
                )
            child_names.append(name)
            deps = list(child.get("dependencies") or [])
            if begin_name not in deps:
                deps.append(begin_name)
            child["dependencies"] = deps
            spec = {**child.get("spec", {})}
            spec.setdefault("on_error", agg_name)
            child["spec"] = spec
            wrapped_children.append(child)

        # 3. Aggregator: did any wrapped child error?  Depends on begin + every
        #    child (≥2 deps) so upstream outputs are namespaced, not flattened.
        failed_terms = " or ".join(f"('_error' in default({n}, {{}}))" for n in child_names)
        agg = {
            "kind": "expression_node",
            "metadata": {"name": agg_name},
            "spec": {
                "expressions": {"any_failed": failed_terms or "False"},
                "output_fields": ["any_failed"],
            },
            "dependencies": [begin_name, *child_names],
        }

        # 4. Release on success / rollback on failure (mutually-exclusive gates).
        release = _savepoint_node(
            release_name, "release_savepoint", when=f"not {agg_name}.any_failed"
        )
        release["dependencies"] = [agg_name]
        rollback = _savepoint_node(
            rollback_name, "rollback_savepoint", when=f"{agg_name}.any_failed"
        )
        rollback["dependencies"] = [agg_name]

        return node_builder([begin, *wrapped_children, agg, release, rollback])
