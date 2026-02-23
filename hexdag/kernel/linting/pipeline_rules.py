"""Pipeline-level lint rules for hexDAG YAML pipelines."""

from __future__ import annotations

import re
from typing import Any

from hexdag.compiler.yaml_validator import _get_known_node_types
from hexdag.kernel.domain.dag import DirectedGraph
from hexdag.kernel.linting.models import LintReport, LintViolation
from hexdag.kernel.linting.rules import LintRule, run_rules
from hexdag.kernel.resolver import get_registered_aliases

# Node types that involve LLM calls (should have timeout/retry)
_LLM_NODE_TYPES = frozenset({
    "llm",
    "llm_node",
    "agent",
    "agent_node",
})

# Regex patterns that suggest hardcoded secrets
_SECRET_PATTERNS = (
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),  # OpenAI keys
    re.compile(r"key-[a-zA-Z0-9]{20,}"),  # Generic API keys
    re.compile(r"ghp_[a-zA-Z0-9]{36,}"),  # GitHub PATs
    re.compile(r"gho_[a-zA-Z0-9]{36,}"),  # GitHub OAuth
    re.compile(r"xox[bpors]-[a-zA-Z0-9-]+"),  # Slack tokens
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key IDs
)

_SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")


def _extract_nodes(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the nodes list from a pipeline config."""
    nodes: list[dict[str, Any]] = config.get("spec", {}).get("nodes", [])
    return nodes


def _node_name(node: dict[str, Any]) -> str:
    """Get the name of a node, falling back to 'unknown'."""
    name: str = node.get("metadata", {}).get("name", "unknown")
    return name


def _node_kind(node: dict[str, Any]) -> str:
    """Get the kind of a node, normalized (without _node suffix)."""
    kind: str = node.get("kind", "")
    if ":" in kind:
        kind = kind.split(":", 1)[1]
    return kind


# ---------------------------------------------------------------------------
# Error rules (E1xx)
# ---------------------------------------------------------------------------


class CycleDetectionRule:
    """E100: Detect cycles in the dependency graph."""

    rule_id = "E100"
    severity = "error"
    description = "Cycle detected in dependency graph"

    def check(self, config: dict[str, Any]) -> list[LintViolation]:
        """Check for cycles in the pipeline dependency graph."""
        nodes = _extract_nodes(config)
        dep_graph: dict[str, set[str]] = {}
        for node in nodes:
            name = _node_name(node)
            deps = node.get("spec", {}).get("dependencies", [])
            if not isinstance(deps, list):
                deps = [deps]
            dep_graph[name] = set(deps)

        if cycle_msg := DirectedGraph.detect_cycle(dep_graph):
            return [
                LintViolation(
                    rule_id=self.rule_id,
                    severity="error",
                    message=cycle_msg,
                    location="dependency graph",
                )
            ]
        return []


class UnresolvableKindRule:
    """E101: Node kind cannot be resolved to a known type."""

    rule_id = "E101"
    severity = "error"
    description = "Unresolvable component kind"

    def check(self, config: dict[str, Any]) -> list[LintViolation]:
        """Check that all node kinds can be resolved."""
        known = _get_known_node_types()
        registered = get_registered_aliases()
        violations: list[LintViolation] = []

        for node in _extract_nodes(config):
            kind = node.get("kind", "")
            name = _node_name(node)

            # Skip macro invocations and full module paths
            if kind == "macro_invocation" or "." in kind:
                continue

            # Skip registered user aliases
            if kind in registered:
                continue

            # Normalize
            raw = kind
            if ":" in raw:
                raw = raw.split(":", 1)[1]
            base = raw[:-5] if raw.endswith("_node") else raw

            if base not in known and raw not in known and kind not in known:
                violations.append(
                    LintViolation(
                        rule_id=self.rule_id,
                        severity="error",
                        message=f"Unknown node kind '{kind}'",
                        location=f"node '{name}'",
                        suggestion=(
                            "Check spelling or use a full module path"
                            " (e.g. hexdag.stdlib.nodes.LLMNode)"
                        ),
                    )
                )

        return violations


# ---------------------------------------------------------------------------
# Warning rules (W2xx)
# ---------------------------------------------------------------------------


class MissingTimeoutRule:
    """W200: LLM/agent nodes should have a timeout configured."""

    rule_id = "W200"
    severity = "warning"
    description = "Missing timeout on LLM/agent node"

    def check(self, config: dict[str, Any]) -> list[LintViolation]:
        """Check that LLM/agent nodes have timeout configured."""
        violations: list[LintViolation] = []
        for node in _extract_nodes(config):
            kind = _node_kind(node)
            if kind not in _LLM_NODE_TYPES:
                continue
            spec = node.get("spec", {})
            if "timeout" not in spec:
                violations.append(
                    LintViolation(
                        rule_id=self.rule_id,
                        severity="warning",
                        message="No timeout configured",
                        location=f"node '{_node_name(node)}'",
                        suggestion="Add 'timeout: 30' (seconds) to the node spec",
                    )
                )
        return violations


class MissingRetryRule:
    """W201: LLM nodes should have retry configuration."""

    rule_id = "W201"
    severity = "warning"
    description = "Missing retry config on LLM node"

    def check(self, config: dict[str, Any]) -> list[LintViolation]:
        """Check that LLM nodes have retry configuration."""
        violations: list[LintViolation] = []
        for node in _extract_nodes(config):
            kind = _node_kind(node)
            if kind not in _LLM_NODE_TYPES:
                continue
            spec = node.get("spec", {})
            if "max_retries" not in spec and "retry" not in spec:
                violations.append(
                    LintViolation(
                        rule_id=self.rule_id,
                        severity="warning",
                        message="No retry configuration",
                        location=f"node '{_node_name(node)}'",
                        suggestion="Add 'max_retries: 3' to handle transient API failures",
                    )
                )
        return violations


class UnusedNodeOutputRule:
    """W202: Node output is never consumed by any downstream node."""

    rule_id = "W202"
    severity = "warning"
    description = "Unused node output (no downstream consumers)"

    def check(self, config: dict[str, Any]) -> list[LintViolation]:
        """Check for nodes whose output is never consumed."""
        nodes = _extract_nodes(config)
        if len(nodes) <= 1:
            return []

        all_names = [_node_name(n) for n in nodes]
        consumed: set[str] = set()

        for node in nodes:
            deps = node.get("spec", {}).get("dependencies", [])
            if not isinstance(deps, list):
                deps = [deps]
            consumed.update(deps)

        # The last node in the list is typically the terminal output — don't flag it.
        terminal = all_names[-1] if all_names else None

        return [
            LintViolation(
                rule_id=self.rule_id,
                severity="warning",
                message="Node output is never used by any downstream node",
                location=f"node '{name}'",
                suggestion="Add this node as a dependency of a downstream node, or remove it",
            )
            for name in all_names
            if name != terminal and name not in consumed
        ]


class HardcodedSecretRule:
    """W203: Config values that look like hardcoded secrets."""

    rule_id = "W203"
    severity = "warning"
    description = "Possible hardcoded secret in config"

    def check(self, config: dict[str, Any]) -> list[LintViolation]:
        """Check for config values that look like hardcoded secrets."""
        violations: list[LintViolation] = []
        self._scan_dict(config, "", violations)
        return violations

    def _scan_dict(self, obj: Any, path: str, violations: list[LintViolation]) -> None:
        """Recursively scan a dict for values matching secret patterns."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                self._scan_dict(value, f"{path}.{key}" if path else key, violations)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                self._scan_dict(item, f"{path}[{i}]", violations)
        elif isinstance(obj, str):
            for pattern in _SECRET_PATTERNS:
                if pattern.search(obj):
                    violations.append(
                        LintViolation(
                            rule_id=self.rule_id,
                            severity="warning",
                            message="Value looks like a hardcoded secret",
                            location=path,
                            suggestion="Use environment variables or a secret manager instead",
                        )
                    )
                    break


# ---------------------------------------------------------------------------
# Info rules (I3xx)
# ---------------------------------------------------------------------------


class MissingDescriptionRule:
    """I300: Node metadata is missing a description."""

    rule_id = "I300"
    severity = "info"
    description = "Missing node description in metadata"

    def check(self, config: dict[str, Any]) -> list[LintViolation]:
        """Check that nodes have description in metadata."""
        violations: list[LintViolation] = []
        for node in _extract_nodes(config):
            metadata = node.get("metadata", {})
            if "description" not in metadata:
                violations.append(
                    LintViolation(
                        rule_id=self.rule_id,
                        severity="info",
                        message="No description in metadata",
                        location=f"node '{_node_name(node)}'",
                        suggestion="Add 'description' to metadata for better documentation",
                    )
                )
        return violations


class NodeNamingRule:
    """I301: Node names should follow snake_case convention."""

    rule_id = "I301"
    severity = "info"
    description = "Node name is not snake_case"

    def check(self, config: dict[str, Any]) -> list[LintViolation]:
        """Check that node names follow snake_case convention."""
        violations: list[LintViolation] = []
        for node in _extract_nodes(config):
            name = _node_name(node)
            if name == "unknown":
                continue
            if not _SNAKE_CASE_RE.match(name):
                violations.append(
                    LintViolation(
                        rule_id=self.rule_id,
                        severity="info",
                        message=f"Node name '{name}' is not snake_case",
                        location=f"node '{name}'",
                        suggestion="Rename to snake_case for consistency",
                    )
                )
        return violations


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

ALL_PIPELINE_RULES: list[LintRule] = [
    CycleDetectionRule(),
    UnresolvableKindRule(),
    MissingTimeoutRule(),
    MissingRetryRule(),
    UnusedNodeOutputRule(),
    HardcodedSecretRule(),
    MissingDescriptionRule(),
    NodeNamingRule(),
]


def run_pipeline_rules(config: dict[str, Any]) -> LintReport:
    """Run all pipeline lint rules and return a report.

    Parameters
    ----------
    config : dict[str, Any]
        Parsed YAML pipeline configuration

    Returns
    -------
    LintReport
        Report containing all violations found
    """
    return run_rules(ALL_PIPELINE_RULES, config)
