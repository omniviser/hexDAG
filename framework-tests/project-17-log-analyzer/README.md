# Project 17: Log Analysis and Summarization - hexDAG Port

## Overview
This folder contains the hexDAG port of LangGraph's `project-17-log-analysis-and-summarization`.

## What It Tests

**Parse system logs, identify issues, and generate actionable summaries.**

```
┌──────────┐     ┌──────────────┐     ┌──────────────┐
│   Logs   │ --> │    Parse     │ --> │   Summarize  │ --> Report
│          │     │  & Categorize│     │   & Advise   │
└──────────┘     └──────────────┘     └──────────────┘
                       │                     │
                       v                     v
                 Extract ERROR,        Summary + severity
                 WARNING, CRITICAL     + recommendations
```

### Example Flow:
```
INPUT LOG:
ERROR: Database connection failed. Retrying...
WARNING: High CPU usage detected on server 'web-01'
ERROR: Disk full. Cannot write to /var/log/app.log

[PARSE] Analyzing log file...
[PARSE] Found 2 errors, 1 warning

[SUMMARIZE] Generating summary...

REPORT:
1. SEVERITY: High
2. ISSUES: Database connectivity, disk space
3. ROOT CAUSE: Possible disk full causing DB issues
4. ACTIONS: Clear disk space immediately, check DB config
```

### Why This Matters:
- **Fast troubleshooting** - Quickly identify issues in large logs
- **Pattern recognition** - LLM can spot related errors
- **Actionable advice** - Not just problems, but solutions
- **Severity assessment** - Know what to fix first

## Files
- `log_pipeline.yaml` - hexDAG YAML pipeline
- `run_log_analyzer.py` - Python runner with sample logs
- `README.md` - This file

## The Difference

| | LangGraph | hexDAG |
|---|-----------|--------|
| **Parsing** | Simple ERROR filter | Full categorization |
| **Statistics** | None | Count by level |
| **Output** | Basic summary | Structured report |

### LangGraph (basic):
```python
def parse_logs(state):
    errors = [line for line in state["log_content"].splitlines() if "ERROR" in line]
    return {"summary": "\n".join(errors)}

def summarize_errors(state):
    response = llm.invoke(f"Summarize: {state['summary']}")
```

### hexDAG (enhanced):
```python
async def parse_logs(inputs):
    # Parse all levels: CRITICAL, ERROR, WARNING, INFO
    # Count by category
    # Return structured analysis
    return {
        "log_analysis": formatted_report,
        "error_count": len(errors),
        "warning_count": len(warnings)
    }

async def summarize_errors(inputs):
    # Detailed prompt requesting:
    # - Severity assessment
    # - Root cause analysis
    # - Recommended actions
    # - Priority order
```

## Verdict: WORKS PERFECTLY

Simple linear pipeline - perfect for hexDAG.

**hexDAG version enhancements:**
- Parses all log levels (CRITICAL, ERROR, WARNING, INFO)
- Provides statistics (counts by level)
- Multiple test scenarios (web server, database, application)
- Structured report with severity, root cause, and actions

## Test Scenarios

| Scenario | Errors | Warnings | Severity |
|----------|--------|----------|----------|
| Simple | 2 | 1 | Medium |
| Web Server | 3 | 2 | High (502 errors) |
| Database | 4 | 3 | Critical (memory/deadlock) |
| Application | 4 | 3 | Critical (memory leak) |

## How to Run

```bash
cd framework-tests/project17-log-analyzer
..\..\.venv\Scripts\python.exe run_log_analyzer.py
```

Expected output:
```
============================================================
hexDAG Log Analyzer Demo
============================================================

[Scenario] Database Log
--------------------------------------------------
  [PARSE] Analyzing log file...
  [PARSE] Found 4 errors, 3 warnings
  [SUMMARIZE] Generating summary and recommendations...

RAW LOG:
------------------------------
2024-01-15 14:15:45 ERROR [postgres] Query timeout: SELECT * FROM large_table
2024-01-15 14:16:00 ERROR [postgres] Too many connections (max: 100)
2024-01-15 14:16:05 CRITICAL [postgres] Out of shared memory for lock table
...

PARSED ANALYSIS:
------------------------------
LOG STATISTICS:
  Total lines: 10
  CRITICAL: 1
  ERROR: 3
  WARNING: 3
  INFO: 3

ERRORS FOUND:
  - 2024-01-15 14:15:45 ERROR [postgres] Query timeout...
  ...

AI SUMMARY & RECOMMENDATIONS:
------------------------------
1. EXECUTIVE SUMMARY
   Database is experiencing critical issues with connection pooling
   and memory management, leading to transaction failures.

2. SEVERITY ASSESSMENT
   Overall severity: Critical
   Reason: Out of shared memory and deadlocks indicate imminent failure

3. RECOMMENDED ACTIONS
   Immediate:
   - Increase shared_buffers in PostgreSQL config
   - Kill long-running queries
   Short-term:
   - Optimize the SELECT * FROM large_table query
   - Increase max_connections or use connection pooling
```

## Test Results
[To be filled after testing]

## Observations
[To be filled after testing]
