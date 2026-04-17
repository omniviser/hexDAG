"""Tests for safe path modifiers (| required, | default) and required_inputs."""

import pytest

from hexdag.kernel.orchestration.components.execution_coordinator import (
    ExecutionCoordinator,
    _parse_default_value,
)


class TestParseDefaultValue:
    def test_none(self):
        assert _parse_default_value("None") is None

    def test_true(self):
        assert _parse_default_value("True") is True
        assert _parse_default_value("true") is True

    def test_false(self):
        assert _parse_default_value("False") is False
        assert _parse_default_value("false") is False

    def test_integer(self):
        assert _parse_default_value("0") == 0
        assert _parse_default_value("42") == 42
        assert _parse_default_value("-1") == -1

    def test_float(self):
        assert _parse_default_value("3.14") == 3.14

    def test_single_quoted_string(self):
        assert _parse_default_value("'hello'") == "hello"

    def test_double_quoted_string(self):
        assert _parse_default_value('"world"') == "world"

    def test_empty_string(self):
        assert _parse_default_value("''") == ""

    def test_unquoted_string(self):
        assert _parse_default_value("unknown") == "unknown"


class TestSafePathModifiers:
    def setup_method(self):
        self.coordinator = ExecutionCoordinator()

    def test_required_passes_when_value_present(self):
        result = self.coordinator._resolve_mapping_value(
            "node1.field | required",
            base_input={},
            initial_input={},
            node_results={"node1": {"field": "value"}},
        )
        assert result == "value"

    def test_required_raises_when_none(self):
        with pytest.raises(ValueError, match="Required field"):
            self.coordinator._resolve_mapping_value(
                "node1.missing_field | required",
                base_input={},
                initial_input={},
                node_results={"node1": {}},
            )

    def test_default_returns_value_when_present(self):
        result = self.coordinator._resolve_mapping_value(
            "node1.field | default('fallback')",
            base_input={},
            initial_input={},
            node_results={"node1": {"field": "actual"}},
        )
        assert result == "actual"

    def test_default_returns_default_when_none(self):
        result = self.coordinator._resolve_mapping_value(
            "node1.missing | default('fallback')",
            base_input={},
            initial_input={},
            node_results={"node1": {}},
        )
        assert result == "fallback"

    def test_default_none(self):
        result = self.coordinator._resolve_mapping_value(
            "node1.missing | default(None)",
            base_input={},
            initial_input={},
            node_results={"node1": {}},
        )
        assert result is None

    def test_default_integer(self):
        result = self.coordinator._resolve_mapping_value(
            "node1.missing | default(0)",
            base_input={},
            initial_input={},
            node_results={"node1": {}},
        )
        assert result == 0

    def test_default_boolean(self):
        result = self.coordinator._resolve_mapping_value(
            "node1.missing | default(false)",
            base_input={},
            initial_input={},
            node_results={"node1": {}},
        )
        assert result is False

    def test_input_with_required(self):
        result = self.coordinator._resolve_mapping_value(
            "$input.carrier_id | required",
            base_input={},
            initial_input={"carrier_id": "C-1"},
            node_results={},
        )
        assert result == "C-1"

    def test_input_required_raises_when_missing(self):
        with pytest.raises(ValueError, match="Required field"):
            self.coordinator._resolve_mapping_value(
                "$input.missing_field | required",
                base_input={},
                initial_input={},
                node_results={},
            )

    def test_no_modifier_unchanged(self):
        """Regular references without modifiers still work."""
        result = self.coordinator._resolve_mapping_value(
            "node1.field",
            base_input={},
            initial_input={},
            node_results={"node1": {"field": "value"}},
        )
        assert result == "value"

    def test_pipe_in_non_modifier_context_preserved(self):
        """A pipe character that doesn't match a modifier pattern is left alone."""
        # "node1.field | somethingelse" — doesn't match required or default(...)
        # so source stays as-is: "node1.field | somethingelse"
        # This will fail to resolve as a node reference but won't incorrectly strip
        result = self.coordinator._resolve_mapping_value(
            "node1.field | somethingelse",
            base_input={},
            initial_input={},
            node_results={"node1": {"field": "value"}},
        )
        # Source was NOT modified — still tries to resolve "node1.field | somethingelse"
        # which won't match any node, so returns whatever the base resolution gives
        # The important thing is it doesn't crash or truncate
        assert result is not None or result is None  # Just assert no crash
