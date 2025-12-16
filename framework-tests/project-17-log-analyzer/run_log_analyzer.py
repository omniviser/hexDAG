#!/usr/bin/env python
"""
hexDAG Log Analyzer Demo
Ported from LangGraph project-17-log-analysis-and-summarization

Pattern: Log Analysis Pipeline
- Parse log files to extract errors, warnings, and info
- Categorize and count by severity
- Generate human-readable summary with recommendations

Run with: ..\..\.venv\Scripts\python.exe run_log_analyzer.py
"""
import asyncio
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load environment variables
env_path = project_root / "reference_examples" / "langgraph-tutorials" / ".env"
load_dotenv(env_path)

import google.generativeai as genai
from hexdag.core.domain.dag import DirectedGraph, NodeSpec
from hexdag.core.orchestration.orchestrator import Orchestrator

# Configure Google Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY not found in .env file")
    sys.exit(1)

genai.configure(api_key=GEMINI_API_KEY)


# Sample log files for testing
SAMPLE_LOGS = {
    "simple": """ERROR: Disk full. Cannot write to /var/log/app.log
INFO: User 'admin' logged in from 192.168.1.100
WARNING: High CPU usage detected on server 'web-01'
ERROR: Database connection failed. Retrying...
""",

    "web_server": """2024-01-15 10:23:45 INFO [nginx] Server started on port 80
2024-01-15 10:24:12 INFO [nginx] New connection from 192.168.1.50
2024-01-15 10:24:15 WARNING [nginx] Slow response time: 2.5s for /api/users
2024-01-15 10:25:01 ERROR [nginx] 502 Bad Gateway - upstream server not responding
2024-01-15 10:25:02 ERROR [nginx] Connection refused to backend:8080
2024-01-15 10:25:05 ERROR [nginx] 502 Bad Gateway - upstream server not responding
2024-01-15 10:26:00 WARNING [nginx] High memory usage: 85%
2024-01-15 10:26:30 INFO [nginx] Backend server recovered
2024-01-15 10:27:00 INFO [nginx] Health check passed
""",

    "database": """2024-01-15 14:00:00 INFO [postgres] Database started
2024-01-15 14:00:05 INFO [postgres] Accepting connections
2024-01-15 14:15:23 WARNING [postgres] Long-running query detected (45s)
2024-01-15 14:15:30 WARNING [postgres] Connection pool exhausted, waiting...
2024-01-15 14:15:45 ERROR [postgres] Query timeout: SELECT * FROM large_table
2024-01-15 14:16:00 ERROR [postgres] Too many connections (max: 100, current: 100)
2024-01-15 14:16:05 CRITICAL [postgres] Out of shared memory for lock table
2024-01-15 14:16:10 ERROR [postgres] Transaction aborted due to deadlock
2024-01-15 14:17:00 INFO [postgres] Connections released, pool available
2024-01-15 14:18:00 WARNING [postgres] Disk usage at 90%
""",

    "application": """2024-01-15 09:00:00 INFO [app] Application started v2.3.1
2024-01-15 09:00:01 INFO [app] Connected to database
2024-01-15 09:00:02 INFO [app] Cache initialized
2024-01-15 09:15:30 WARNING [app] API rate limit approaching (80%)
2024-01-15 09:20:45 ERROR [app] Failed to process order #12345: Payment declined
2024-01-15 09:21:00 ERROR [app] Email service unavailable
2024-01-15 09:21:05 WARNING [app] Falling back to queue for email delivery
2024-01-15 09:30:00 ERROR [app] NullPointerException in UserService.getProfile()
2024-01-15 09:30:01 ERROR [app] Stack trace: at com.app.service.UserService.getProfile(UserService.java:45)
2024-01-15 09:45:00 CRITICAL [app] Memory leak detected - heap usage 95%
2024-01-15 09:45:05 WARNING [app] Initiating garbage collection
2024-01-15 09:46:00 INFO [app] GC completed, heap usage now 60%
"""
}


def parse_log_line(line: str) -> dict | None:
    """Parse a single log line and extract components."""
    line = line.strip()
    if not line:
        return None

    # Determine log level
    level = "INFO"
    for lvl in ["CRITICAL", "ERROR", "WARNING", "WARN", "INFO", "DEBUG"]:
        if lvl in line.upper():
            level = lvl
            break

    return {
        "level": level,
        "message": line,
        "is_error": level in ["ERROR", "CRITICAL"],
        "is_warning": level in ["WARNING", "WARN"]
    }


async def parse_logs(inputs: dict) -> dict:
    """
    Parse logs and categorize by severity.

    LangGraph version:
        errors = [line for line in state["log_content"].splitlines() if "ERROR" in line]
        return {"summary": "\n".join(errors)}

    hexDAG version: More comprehensive parsing with statistics.
    """
    log_content = inputs.get("log_content", "")

    print(f"  [PARSE] Analyzing log file...")

    lines = log_content.strip().split('\n')
    parsed_lines = [parse_log_line(line) for line in lines if line.strip()]
    parsed_lines = [p for p in parsed_lines if p]  # Remove None

    # Count by level
    level_counts = Counter(p["level"] for p in parsed_lines)

    # Extract errors and warnings
    errors = [p["message"] for p in parsed_lines if p["is_error"]]
    warnings = [p["message"] for p in parsed_lines if p["is_warning"]]

    # Build analysis report
    analysis_lines = [
        "LOG STATISTICS:",
        f"  Total lines: {len(parsed_lines)}",
        f"  CRITICAL: {level_counts.get('CRITICAL', 0)}",
        f"  ERROR: {level_counts.get('ERROR', 0)}",
        f"  WARNING: {level_counts.get('WARNING', 0) + level_counts.get('WARN', 0)}",
        f"  INFO: {level_counts.get('INFO', 0)}",
        "",
        "ERRORS FOUND:",
    ]

    if errors:
        for error in errors:
            analysis_lines.append(f"  - {error}")
    else:
        analysis_lines.append("  None")

    analysis_lines.append("")
    analysis_lines.append("WARNINGS FOUND:")

    if warnings:
        for warning in warnings:
            analysis_lines.append(f"  - {warning}")
    else:
        analysis_lines.append("  None")

    log_analysis = "\n".join(analysis_lines)

    print(f"  [PARSE] Found {len(errors)} errors, {len(warnings)} warnings")

    return {
        "log_analysis": log_analysis,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "total_lines": len(parsed_lines)
    }


async def summarize_errors(inputs: dict) -> dict:
    """
    Generate summary and recommendations.

    LangGraph version:
        response = llm.invoke(f"Summarize the following error logs:\n\n{state['summary']}")

    hexDAG version: Detailed analysis with severity and recommendations.
    """
    log_analysis = inputs.get("log_analysis", "")
    error_count = inputs.get("error_count", 0)
    warning_count = inputs.get("warning_count", 0)

    print(f"  [SUMMARIZE] Generating summary and recommendations...")

    prompt = f"""You are an experienced DevOps engineer analyzing system logs.

{log_analysis}

Provide a clear, actionable report:

1. EXECUTIVE SUMMARY
   - Brief overview of system health (1-2 sentences)

2. SEVERITY ASSESSMENT
   - Overall severity: Critical / High / Medium / Low
   - Justify your assessment

3. ISSUES IDENTIFIED
   - List each distinct issue found
   - Group related errors together

4. ROOT CAUSE ANALYSIS
   - Potential causes for the errors
   - Any patterns you notice

5. RECOMMENDED ACTIONS
   - Immediate actions (if critical)
   - Short-term fixes
   - Long-term improvements

6. PRIORITY ORDER
   - Which issues to address first and why

Be concise but thorough. Focus on actionable insights."""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    summary = response.text.strip()
    print(f"  [SUMMARIZE] Report generated")

    return {
        "summary": summary,
        "error_count": error_count,
        "warning_count": warning_count
    }


async def run_log_analyzer_demo():
    """
    Demonstrate log analysis with different log types.
    """
    print("=" * 60)
    print("hexDAG Log Analyzer Demo")
    print("=" * 60)
    print()

    scenarios = [
        ("Simple Log", "simple"),
        ("Web Server Log", "web_server"),
        ("Database Log", "database"),
        ("Application Log", "application"),
    ]

    for scenario_name, scenario_key in scenarios:
        print(f"[Scenario] {scenario_name}")
        print("-" * 50)

        log_content = SAMPLE_LOGS[scenario_key]

        # Parse logs
        parse_result = await parse_logs({"log_content": log_content})

        # Generate summary
        summarize_input = {
            "log_analysis": parse_result["log_analysis"],
            "error_count": parse_result["error_count"],
            "warning_count": parse_result["warning_count"]
        }
        summary_result = await summarize_errors(summarize_input)

        # Display
        print()
        print("RAW LOG:")
        print("-" * 30)
        print(log_content.strip())
        print()
        print("PARSED ANALYSIS:")
        print("-" * 30)
        print(parse_result["log_analysis"])
        print()
        print("AI SUMMARY & RECOMMENDATIONS:")
        print("-" * 30)
        print(summary_result["summary"])
        print()
        print("=" * 60)
        print()


async def analyze_log_file(log_content: str):
    """
    Analyze a single log file.
    """
    print("Analyzing logs...")
    print("-" * 50)

    parse_result = await parse_logs({"log_content": log_content})
    summary_result = await summarize_errors({
        "log_analysis": parse_result["log_analysis"],
        "error_count": parse_result["error_count"],
        "warning_count": parse_result["warning_count"]
    })

    print()
    print("REPORT:")
    print("=" * 50)
    print(summary_result["summary"])

    return summary_result


async def main():
    """Main entry point."""
    await run_log_analyzer_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
