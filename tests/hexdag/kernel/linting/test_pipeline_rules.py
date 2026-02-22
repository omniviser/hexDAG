"""Tests for hexdag.kernel.linting.pipeline_rules."""

from __future__ import annotations

from hexdag.kernel.linting.pipeline_rules import (
    CycleDetectionRule,
    HardcodedSecretRule,
    MissingDescriptionRule,
    MissingRetryRule,
    MissingTimeoutRule,
    NodeNamingRule,
    UnresolvableKindRule,
    UnusedNodeOutputRule,
    run_pipeline_rules,
)


def _pipeline(*nodes: dict) -> dict:
    """Helper to build a minimal pipeline config."""
    return {
        "apiVersion": "hexdag/v1",
        "kind": "Pipeline",
        "metadata": {"name": "test"},
        "spec": {"nodes": list(nodes)},
    }


def _node(
    name: str,
    kind: str = "llm_node",
    deps: list[str] | None = None,
    **spec_extra: object,
) -> dict:
    """Helper to build a minimal node config."""
    spec: dict = {"dependencies": deps or []}
    spec.update(spec_extra)
    return {"kind": kind, "metadata": {"name": name}, "spec": spec}


# ---------------------------------------------------------------------------
# E100: Cycle detection
# ---------------------------------------------------------------------------


class TestCycleDetectionRule:
    rule = CycleDetectionRule()

    def test_no_cycle(self) -> None:
        config = _pipeline(
            _node("a", deps=[]),
            _node("b", deps=["a"]),
        )
        assert self.rule.check(config) == []

    def test_cycle_detected(self) -> None:
        config = _pipeline(
            _node("a", deps=["b"]),
            _node("b", deps=["a"]),
        )
        violations = self.rule.check(config)
        assert len(violations) == 1
        assert violations[0].rule_id == "E100"
        assert violations[0].severity == "error"

    def test_self_cycle(self) -> None:
        config = _pipeline(_node("a", deps=["a"]))
        violations = self.rule.check(config)
        assert len(violations) == 1


# ---------------------------------------------------------------------------
# E101: Unresolvable kind
# ---------------------------------------------------------------------------


class TestUnresolvableKindRule:
    rule = UnresolvableKindRule()

    def test_known_kind(self) -> None:
        config = _pipeline(_node("a", kind="llm_node"))
        assert self.rule.check(config) == []

    def test_unknown_kind(self) -> None:
        config = _pipeline(_node("a", kind="nonexistent_type"))
        violations = self.rule.check(config)
        assert len(violations) == 1
        assert violations[0].rule_id == "E101"

    def test_full_module_path_skipped(self) -> None:
        config = _pipeline(_node("a", kind="hexdag.stdlib.nodes.LLMNode"))
        assert self.rule.check(config) == []

    def test_macro_invocation_skipped(self) -> None:
        config = _pipeline(_node("a", kind="macro_invocation"))
        assert self.rule.check(config) == []


# ---------------------------------------------------------------------------
# W200: Missing timeout
# ---------------------------------------------------------------------------


class TestMissingTimeoutRule:
    rule = MissingTimeoutRule()

    def test_llm_without_timeout(self) -> None:
        config = _pipeline(_node("a", kind="llm_node"))
        violations = self.rule.check(config)
        assert len(violations) == 1
        assert violations[0].rule_id == "W200"

    def test_llm_with_timeout(self) -> None:
        config = _pipeline(_node("a", kind="llm_node", timeout=30))
        assert self.rule.check(config) == []

    def test_agent_without_timeout(self) -> None:
        config = _pipeline(_node("a", kind="agent_node"))
        violations = self.rule.check(config)
        assert len(violations) == 1

    def test_function_node_ignored(self) -> None:
        config = _pipeline(_node("a", kind="function_node"))
        assert self.rule.check(config) == []


# ---------------------------------------------------------------------------
# W201: Missing retry
# ---------------------------------------------------------------------------


class TestMissingRetryRule:
    rule = MissingRetryRule()

    def test_llm_without_retry(self) -> None:
        config = _pipeline(_node("a", kind="llm_node"))
        violations = self.rule.check(config)
        assert len(violations) == 1
        assert violations[0].rule_id == "W201"

    def test_llm_with_max_retries(self) -> None:
        config = _pipeline(_node("a", kind="llm_node", max_retries=3))
        assert self.rule.check(config) == []

    def test_llm_with_retry(self) -> None:
        config = _pipeline(_node("a", kind="llm_node", retry=True))
        assert self.rule.check(config) == []

    def test_function_node_ignored(self) -> None:
        config = _pipeline(_node("a", kind="function_node"))
        assert self.rule.check(config) == []


# ---------------------------------------------------------------------------
# W202: Unused node output
# ---------------------------------------------------------------------------


class TestUnusedNodeOutputRule:
    rule = UnusedNodeOutputRule()

    def test_all_used(self) -> None:
        config = _pipeline(
            _node("a"),
            _node("b", deps=["a"]),
        )
        assert self.rule.check(config) == []

    def test_unused_non_terminal(self) -> None:
        config = _pipeline(
            _node("a"),
            _node("b"),
            _node("c", deps=["b"]),
        )
        violations = self.rule.check(config)
        assert len(violations) == 1
        assert violations[0].rule_id == "W202"
        assert "a" in violations[0].location

    def test_single_node_no_violation(self) -> None:
        config = _pipeline(_node("a"))
        assert self.rule.check(config) == []

    def test_terminal_node_not_flagged(self) -> None:
        """Last node in list should not be flagged even if unused."""
        config = _pipeline(
            _node("a"),
            _node("b"),
        )
        violations = self.rule.check(config)
        # "a" is unused and not terminal, "b" is terminal
        assert len(violations) == 1
        assert "a" in violations[0].location


# ---------------------------------------------------------------------------
# W203: Hardcoded secrets
# ---------------------------------------------------------------------------


class TestHardcodedSecretRule:
    rule = HardcodedSecretRule()

    def test_clean_config(self) -> None:
        config = _pipeline(_node("a"))
        assert self.rule.check(config) == []

    def test_openai_key_detected(self) -> None:
        config = _pipeline(_node("a", api_key="sk-abcdefghijklmnopqrstuvwxyz12345678"))
        violations = self.rule.check(config)
        assert len(violations) == 1
        assert violations[0].rule_id == "W203"

    def test_aws_key_detected(self) -> None:
        config = {"spec": {"nodes": []}, "ports": {"config": {"key": "AKIAIOSFODNN7EXAMPLE"}}}
        violations = self.rule.check(config)
        assert len(violations) == 1

    def test_env_var_reference_clean(self) -> None:
        config = _pipeline(_node("a", api_key="${OPENAI_API_KEY}"))
        assert self.rule.check(config) == []


# ---------------------------------------------------------------------------
# I300: Missing description
# ---------------------------------------------------------------------------


class TestMissingDescriptionRule:
    rule = MissingDescriptionRule()

    def test_no_description(self) -> None:
        config = _pipeline(_node("a"))
        violations = self.rule.check(config)
        assert len(violations) == 1
        assert violations[0].rule_id == "I300"

    def test_with_description(self) -> None:
        node = _node("a")
        node["metadata"]["description"] = "Analyzes input"
        config = _pipeline(node)
        assert self.rule.check(config) == []


# ---------------------------------------------------------------------------
# I301: Node naming
# ---------------------------------------------------------------------------


class TestNodeNamingRule:
    rule = NodeNamingRule()

    def test_snake_case(self) -> None:
        config = _pipeline(_node("my_node"))
        assert self.rule.check(config) == []

    def test_camel_case(self) -> None:
        config = _pipeline(_node("myNode"))
        violations = self.rule.check(config)
        assert len(violations) == 1
        assert violations[0].rule_id == "I301"

    def test_kebab_case(self) -> None:
        config = _pipeline(_node("my-node"))
        violations = self.rule.check(config)
        assert len(violations) == 1

    def test_uppercase(self) -> None:
        config = _pipeline(_node("MyNode"))
        violations = self.rule.check(config)
        assert len(violations) == 1

    def test_single_word(self) -> None:
        config = _pipeline(_node("analyzer"))
        assert self.rule.check(config) == []


# ---------------------------------------------------------------------------
# Integration: run_pipeline_rules
# ---------------------------------------------------------------------------


class TestRunPipelineRules:
    def test_clean_pipeline(self) -> None:
        node = _node("analyzer", kind="function_node")
        node["metadata"]["description"] = "Does stuff"
        config = _pipeline(node)
        report = run_pipeline_rules(config)
        assert report.is_clean

    def test_multiple_violations(self) -> None:
        config = _pipeline(
            _node("MyBadNode", kind="llm_node"),
        )
        report = run_pipeline_rules(config)
        assert not report.is_clean
        rule_ids = {v.rule_id for v in report.violations}
        # Should have at least: W200 (timeout), W201 (retry), I300 (desc), I301 (naming)
        assert "W200" in rule_ids
        assert "W201" in rule_ids
        assert "I300" in rule_ids
        assert "I301" in rule_ids

    def test_empty_pipeline(self) -> None:
        config = _pipeline()
        report = run_pipeline_rules(config)
        assert report.is_clean
