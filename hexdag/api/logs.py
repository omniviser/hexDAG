"""Log query API — MCP/Studio tools for querying hexDAG JSON log files.

Provides unified functions that both the MCP server and hexdag-studio
REST API consume. Reads Loguru JSON log files (written by
``configure_logging(output_file=...)``) and filters on structured fields.

MCP server usage::

    from hexdag.api import logs

    @mcp.tool()
    async def query_logs(
        level: str | None = None,
        pipeline_name: str | None = None,
        contains: str | None = None,
    ):
        return await logs.query_logs(
            "hexdag.log", level=level, pipeline_name=pipeline_name, contains=contains,
        )

Studio REST API::

    @router.get("/logs")
    async def get_logs(level: str | None = None, pipeline_name: str | None = None):
        return await logs.query_logs("hexdag.log", level=level, pipeline_name=pipeline_name)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import aiofiles

_LEVEL_ORDER = {
    "TRACE": 0,
    "DEBUG": 1,
    "INFO": 2,
    "WARNING": 3,
    "ERROR": 4,
    "CRITICAL": 5,
}


async def query_logs(
    log_file: str | Path,
    *,
    level: str | None = None,
    pipeline_name: str | None = None,
    node_id: str | None = None,
    contains: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Query log records from a JSON log file with filters.

    Each line in the log file is a JSON record produced by Loguru's
    ``serialize=True`` mode. Structured fields (``pipeline_name``,
    ``node``, ``node_type``) are extracted from ``record.extra``.

    Args
    ----
        log_file: Path to the JSON log file.
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        pipeline_name: Filter by pipeline name (exact match on extra.pipeline_name).
        node_id: Filter by node name (exact match on extra.node).
        contains: Substring match on the log message text.
        limit: Maximum number of records to return (default 100).

    Returns
    -------
        List of matching log records, newest first.
    """
    path = Path(log_file)
    if not path.exists():
        return []

    min_level = _LEVEL_ORDER.get(level.upper(), 0) if level else 0
    contains_lower = contains.lower() if contains else None

    records = await _read_json_lines(path)

    results: list[dict[str, Any]] = []
    for rec in reversed(records):
        if len(results) >= limit:
            break

        record = rec.get("record", {})
        extra = record.get("extra", {})
        rec_level = record.get("level", {}).get("name", "INFO")

        # Level filter
        if _LEVEL_ORDER.get(rec_level, 0) < min_level:
            continue

        # Pipeline filter
        if pipeline_name and extra.get("pipeline_name") != pipeline_name:
            continue

        # Node filter
        if node_id and extra.get("node") != node_id:
            continue

        # Substring match
        message = record.get("message", rec.get("text", ""))
        if contains_lower and contains_lower not in message.lower():
            continue

        results.append(_summarize_record(rec))

    return results


async def get_log_summary(
    log_file: str | Path,
    *,
    pipeline_name: str | None = None,
) -> dict[str, Any]:
    """Aggregate log counts by level from a JSON log file.

    Args
    ----
        log_file: Path to the JSON log file.
        pipeline_name: Scope summary to a specific pipeline.

    Returns
    -------
        Dict with counts per level, total count, and time range.
    """
    path = Path(log_file)
    if not path.exists():
        return {"total": 0, "counts": {}, "pipeline_name": pipeline_name}

    records = await _read_json_lines(path)

    counts: dict[str, int] = {}
    total = 0

    for rec in records:
        record = rec.get("record", {})
        extra = record.get("extra", {})

        if pipeline_name and extra.get("pipeline_name") != pipeline_name:
            continue

        rec_level = record.get("level", {}).get("name", "UNKNOWN")
        counts[rec_level] = counts.get(rec_level, 0) + 1
        total += 1

    return {
        "total": total,
        "counts": counts,
        "pipeline_name": pipeline_name,
    }


async def tail_logs(
    log_file: str | Path,
    *,
    n: int = 20,
    level: str | None = None,
) -> list[dict[str, Any]]:
    """Get the N most recent log records from a JSON log file.

    Args
    ----
        log_file: Path to the JSON log file.
        n: Number of records to return (default 20).
        level: Minimum log level filter.

    Returns
    -------
        List of most recent records, newest first.
    """
    return await query_logs(log_file, level=level, limit=n)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    """Read a JSON-lines log file, skipping malformed lines."""
    records: list[dict[str, Any]] = []
    async with aiofiles.open(path, encoding="utf-8") as f:
        async for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _summarize_record(rec: dict[str, Any]) -> dict[str, Any]:
    """Extract key fields from a Loguru JSON record for API output."""
    record = rec.get("record", {})
    extra = record.get("extra", {})
    level_info = record.get("level", {})
    time_info = record.get("time", {})

    result: dict[str, Any] = {
        "timestamp": time_info.get("repr", ""),
        "level": level_info.get("name", ""),
        "message": record.get("message", rec.get("text", "")),
        "module": extra.get("module", record.get("name", "")),
        "function": record.get("function", ""),
        "line": record.get("line", 0),
    }

    if pipeline_name := extra.get("pipeline_name"):
        result["pipeline_name"] = pipeline_name
    if node := extra.get("node"):
        result["node"] = node
    if node_type := extra.get("node_type"):
        result["node_type"] = node_type

    return result
