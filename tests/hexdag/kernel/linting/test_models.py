"""Tests for hexdag.kernel.linting.models."""

from hexdag.kernel.linting.models import LintReport, LintViolation


class TestLintViolation:
    def test_creation(self) -> None:
        v = LintViolation(
            rule_id="E100",
            severity="error",
            message="Cycle detected",
            location="dependency graph",
            suggestion="Remove the cycle",
        )
        assert v.rule_id == "E100"
        assert v.severity == "error"
        assert v.message == "Cycle detected"
        assert v.location == "dependency graph"
        assert v.suggestion == "Remove the cycle"

    def test_defaults(self) -> None:
        v = LintViolation(rule_id="W200", severity="warning", message="test")
        assert v.location == ""
        assert v.suggestion is None

    def test_frozen(self) -> None:
        v = LintViolation(rule_id="I300", severity="info", message="test")
        try:
            v.rule_id = "X"  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


class TestLintReport:
    def test_empty_report(self) -> None:
        report = LintReport()
        assert report.is_clean
        assert not report.has_errors
        assert report.violations == []
        assert report.errors == []
        assert report.warnings == []
        assert report.info == []

    def test_add_violations(self) -> None:
        report = LintReport()
        report.add(LintViolation(rule_id="E100", severity="error", message="err"))
        report.add(LintViolation(rule_id="W200", severity="warning", message="warn"))
        report.add(LintViolation(rule_id="I300", severity="info", message="info"))

        assert not report.is_clean
        assert report.has_errors
        assert len(report.violations) == 3
        assert len(report.errors) == 1
        assert len(report.warnings) == 1
        assert len(report.info) == 1

    def test_no_errors_means_no_has_errors(self) -> None:
        report = LintReport()
        report.add(LintViolation(rule_id="W200", severity="warning", message="warn"))
        assert not report.is_clean
        assert not report.has_errors
