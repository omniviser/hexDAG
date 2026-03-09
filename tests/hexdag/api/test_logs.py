"""Tests for hexdag.api.logs — log query API functions."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from hexdag.api import logs

if TYPE_CHECKING:
    from pathlib import Path


def _write_json_log(path: Path, records: list[dict]) -> None:
    """Write JSON-lines log file for testing."""
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _make_record(
    message: str,
    level: str = "INFO",
    pipeline_name: str | None = None,
    node: str | None = None,
    node_type: str | None = None,
    timestamp: str = "2026-03-09T12:00:00",
) -> dict:
    """Create a Loguru-style JSON log record."""
    extra: dict = {}
    if pipeline_name:
        extra["pipeline_name"] = pipeline_name
    if node:
        extra["node"] = node
    if node_type:
        extra["node_type"] = node_type

    return {
        "text": message,
        "record": {
            "message": message,
            "level": {"name": level, "no": 20},
            "time": {"repr": timestamp, "timestamp": 0},
            "name": "test",
            "function": "test_fn",
            "line": 42,
            "extra": extra,
        },
    }


# ---------------------------------------------------------------------------
# query_logs
# ---------------------------------------------------------------------------


class TestQueryLogs:
    """Test query_logs function."""

    @pytest.mark.asyncio()
    async def test_returns_empty_for_missing_file(self, tmp_path: Path) -> None:
        result = await logs.query_logs(tmp_path / "nonexistent.log")
        assert result == []

    @pytest.mark.asyncio()
    async def test_returns_all_records(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        _write_json_log(
            log_file,
            [
                _make_record("msg1"),
                _make_record("msg2"),
                _make_record("msg3"),
            ],
        )
        result = await logs.query_logs(log_file)
        assert len(result) == 3

    @pytest.mark.asyncio()
    async def test_newest_first(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        _write_json_log(
            log_file,
            [
                _make_record("first"),
                _make_record("second"),
                _make_record("third"),
            ],
        )
        result = await logs.query_logs(log_file)
        assert result[0]["message"] == "third"
        assert result[2]["message"] == "first"

    @pytest.mark.asyncio()
    async def test_filter_by_level(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        _write_json_log(
            log_file,
            [
                _make_record("debug msg", level="DEBUG"),
                _make_record("info msg", level="INFO"),
                _make_record("error msg", level="ERROR"),
            ],
        )
        result = await logs.query_logs(log_file, level="WARNING")
        assert len(result) == 1
        assert result[0]["level"] == "ERROR"

    @pytest.mark.asyncio()
    async def test_filter_by_pipeline_name(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        _write_json_log(
            log_file,
            [
                _make_record("p1 msg", pipeline_name="pipeline-a"),
                _make_record("p2 msg", pipeline_name="pipeline-b"),
                _make_record("no pipeline"),
            ],
        )
        result = await logs.query_logs(log_file, pipeline_name="pipeline-a")
        assert len(result) == 1
        assert result[0]["message"] == "p1 msg"
        assert result[0]["pipeline_name"] == "pipeline-a"

    @pytest.mark.asyncio()
    async def test_filter_by_node_id(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        _write_json_log(
            log_file,
            [
                _make_record("node msg", node="analyzer"),
                _make_record("other msg", node="summarizer"),
            ],
        )
        result = await logs.query_logs(log_file, node_id="analyzer")
        assert len(result) == 1
        assert result[0]["node"] == "analyzer"

    @pytest.mark.asyncio()
    async def test_filter_by_contains(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        _write_json_log(
            log_file,
            [
                _make_record("Pipeline started successfully"),
                _make_record("Node completed"),
                _make_record("Pipeline failed with error"),
            ],
        )
        result = await logs.query_logs(log_file, contains="pipeline")
        assert len(result) == 2

    @pytest.mark.asyncio()
    async def test_contains_is_case_insensitive(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        _write_json_log(
            log_file,
            [
                _make_record("ERROR occurred"),
                _make_record("error detected"),
            ],
        )
        result = await logs.query_logs(log_file, contains="ERROR")
        assert len(result) == 2

    @pytest.mark.asyncio()
    async def test_limit_restricts_results(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        _write_json_log(log_file, [_make_record(f"msg{i}") for i in range(20)])
        result = await logs.query_logs(log_file, limit=5)
        assert len(result) == 5

    @pytest.mark.asyncio()
    async def test_combined_filters(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        _write_json_log(
            log_file,
            [
                _make_record("debug pipeline-a", level="DEBUG", pipeline_name="pipeline-a"),
                _make_record("error pipeline-a", level="ERROR", pipeline_name="pipeline-a"),
                _make_record("error pipeline-b", level="ERROR", pipeline_name="pipeline-b"),
            ],
        )
        result = await logs.query_logs(
            log_file,
            level="ERROR",
            pipeline_name="pipeline-a",
        )
        assert len(result) == 1
        assert result[0]["message"] == "error pipeline-a"

    @pytest.mark.asyncio()
    async def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        with log_file.open("w") as f:
            f.write("not json\n")
            f.write(json.dumps(_make_record("valid")) + "\n")
            f.write("also bad {{\n")
        result = await logs.query_logs(log_file)
        assert len(result) == 1
        assert result[0]["message"] == "valid"

    @pytest.mark.asyncio()
    async def test_skips_empty_lines(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        with log_file.open("w") as f:
            f.write(json.dumps(_make_record("msg1")) + "\n")
            f.write("\n")
            f.write("  \n")
            f.write(json.dumps(_make_record("msg2")) + "\n")
        result = await logs.query_logs(log_file)
        assert len(result) == 2

    @pytest.mark.asyncio()
    async def test_summarize_record_fields(self, tmp_path: Path) -> None:
        """Verify that returned records have the expected fields."""
        log_file = tmp_path / "test.log"
        _write_json_log(
            log_file,
            [
                _make_record(
                    "test msg",
                    level="WARNING",
                    pipeline_name="my-pipeline",
                    node="my-node",
                    node_type="llm_node",
                ),
            ],
        )
        result = await logs.query_logs(log_file)
        assert len(result) == 1
        rec = result[0]
        assert rec["timestamp"] == "2026-03-09T12:00:00"
        assert rec["level"] == "WARNING"
        assert rec["message"] == "test msg"
        assert rec["function"] == "test_fn"
        assert rec["line"] == 42
        assert rec["pipeline_name"] == "my-pipeline"
        assert rec["node"] == "my-node"
        assert rec["node_type"] == "llm_node"


# ---------------------------------------------------------------------------
# get_log_summary
# ---------------------------------------------------------------------------


class TestGetLogSummary:
    """Test get_log_summary function."""

    @pytest.mark.asyncio()
    async def test_returns_empty_summary_for_missing_file(self, tmp_path: Path) -> None:
        result = await logs.get_log_summary(tmp_path / "nonexistent.log")
        assert result["total"] == 0
        assert result["counts"] == {}

    @pytest.mark.asyncio()
    async def test_counts_by_level(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        _write_json_log(
            log_file,
            [
                _make_record("m1", level="INFO"),
                _make_record("m2", level="INFO"),
                _make_record("m3", level="ERROR"),
                _make_record("m4", level="DEBUG"),
            ],
        )
        result = await logs.get_log_summary(log_file)
        assert result["total"] == 4
        assert result["counts"]["INFO"] == 2
        assert result["counts"]["ERROR"] == 1
        assert result["counts"]["DEBUG"] == 1

    @pytest.mark.asyncio()
    async def test_filter_by_pipeline_name(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        _write_json_log(
            log_file,
            [
                _make_record("m1", level="INFO", pipeline_name="p1"),
                _make_record("m2", level="ERROR", pipeline_name="p1"),
                _make_record("m3", level="INFO", pipeline_name="p2"),
            ],
        )
        result = await logs.get_log_summary(log_file, pipeline_name="p1")
        assert result["total"] == 2
        assert result["counts"]["INFO"] == 1
        assert result["counts"]["ERROR"] == 1
        assert result["pipeline_name"] == "p1"


# ---------------------------------------------------------------------------
# tail_logs
# ---------------------------------------------------------------------------


class TestTailLogs:
    """Test tail_logs function."""

    @pytest.mark.asyncio()
    async def test_returns_n_most_recent(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        _write_json_log(log_file, [_make_record(f"msg{i}") for i in range(50)])
        result = await logs.tail_logs(log_file, n=10)
        assert len(result) == 10
        # Newest first — last written record is msg49
        assert result[0]["message"] == "msg49"

    @pytest.mark.asyncio()
    async def test_default_n_is_20(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        _write_json_log(log_file, [_make_record(f"msg{i}") for i in range(50)])
        result = await logs.tail_logs(log_file)
        assert len(result) == 20

    @pytest.mark.asyncio()
    async def test_tail_with_level_filter(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        _write_json_log(
            log_file,
            [
                _make_record("debug", level="DEBUG"),
                _make_record("info", level="INFO"),
                _make_record("error", level="ERROR"),
            ],
        )
        result = await logs.tail_logs(log_file, n=10, level="ERROR")
        assert len(result) == 1
        assert result[0]["level"] == "ERROR"

    @pytest.mark.asyncio()
    async def test_returns_empty_for_missing_file(self, tmp_path: Path) -> None:
        result = await logs.tail_logs(tmp_path / "nonexistent.log")
        assert result == []
