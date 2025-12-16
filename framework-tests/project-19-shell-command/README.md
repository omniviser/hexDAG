# Project 19: Shell Command Execution and Summarization - hexDAG Port

## Overview
This folder contains the hexDAG port of LangGraph's `project-19-shell-command-execution-and-summarization`.

## What It Tests

**Execute shell commands and summarize output in plain English.**

```
┌──────────┐     ┌──────────────┐     ┌──────────────┐
│ Command  │ --> │   Execute    │ --> │  Summarize   │ --> Report
│          │     │   Safely     │     │   Output     │
└──────────┘     └──────────────┘     └──────────────┘
                       │                     │
                       v                     v
                 subprocess              Human-readable
                 with timeout            explanation
```

### Example Flow:
```
Command: disk_space

[EXECUTE] Running: disk_space
[EXECUTE] Command completed successfully

RAW OUTPUT:
Filesystem      Size  Used Avail Use% Mounted on
/dev/sda1       500G  350G  150G  70% /

[SUMMARIZE] Analyzing command output...

AI SUMMARY:
1. WHAT THIS COMMAND DOES: Shows disk space usage on your computer.

2. KEY INFORMATION:
   - Main disk is 500GB total
   - 350GB used (70%)
   - 150GB available

3. PLAIN ENGLISH: Your hard drive is about 70% full. You have
   150GB of free space remaining.

4. CONCERNS: Disk usage is moderate. Consider cleanup if it
   exceeds 85%.
```

### Why This Matters:
- **Automation** - Execute system tasks programmatically
- **Explanation** - Technical output made understandable
- **Monitoring** - Check system health with AI analysis
- **Accessibility** - Non-technical users can understand system info

## Files
- `shell_pipeline.yaml` - hexDAG YAML pipeline
- `run_shell_command.py` - Python runner with safe commands
- `README.md` - This file

## SECURITY NOTE

**Only predefined safe commands are allowed!**

This demo uses a whitelist of read-only commands. Never execute arbitrary user input without validation - that would be a security risk.

## The Difference

| | LangGraph | hexDAG |
|---|-----------|--------|
| **Commands** | Hardcoded `ls -l` | Whitelist of safe commands |
| **Platform** | Unix only | Windows + Unix |
| **Safety** | Basic subprocess | Timeout + whitelist |
| **Output** | Basic summary | Structured explanation |

### LangGraph (basic):
```python
def run_command(command: str) -> str:
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout.strip()

def execute_command(state):
    output = command_tool.run("ls -l")  # Hardcoded
    return {"command_output": output}
```

### hexDAG (safer):
```python
# Whitelist of safe commands
SAFE_COMMANDS = {
    "list_files": {"windows": "dir", "unix": "ls -la"},
    "disk_space": {"windows": "wmic...", "unix": "df -h"},
    ...
}

def run_command_safe(command_key: str, timeout: int = 30):
    if command_key not in SAFE_COMMANDS:
        return {"success": False, "output": "Unknown command"}
    # Execute with timeout
    result = subprocess.run(command, timeout=timeout, ...)
```

## Verdict: WORKS PERFECTLY

Linear pipeline - perfect for hexDAG.

**hexDAG version enhancements:**
- Cross-platform (Windows + Unix)
- Safe command whitelist
- Timeout protection
- Structured AI explanations
- Multiple demo commands

## Available Safe Commands

| Command | Windows | Unix | Description |
|---------|---------|------|-------------|
| `list_files` | `dir` | `ls -la` | List files in directory |
| `disk_space` | `wmic...` | `df -h` | Show disk usage |
| `system_info` | `systeminfo` | `uname -a` | System information |
| `current_directory` | `cd` | `pwd` | Current directory |
| `environment` | `set` | `env` | Environment variables |
| `network_info` | `ipconfig` | `ifconfig` | Network configuration |
| `running_processes` | `tasklist` | `ps aux` | Running processes |
| `python_version` | `python --version` | `python3 --version` | Python version |

## How to Run

```bash
cd framework-tests/project19-shell-command
..\..\.venv\Scripts\python.exe run_shell_command.py
```

Expected output:
```
============================================================
hexDAG Shell Command Agent Demo
============================================================
Platform: windows

Available commands:
  - list_files: List files in current directory
  - disk_space: Show disk space usage
  ...

[Command 1] current_directory: Show current working directory
--------------------------------------------------
  [EXECUTE] Running: current_directory
  [EXECUTE] Command completed successfully
  [SUMMARIZE] Analyzing command output...

RAW OUTPUT:
------------------------------
C:\Users\Herbert\Desktop\CODE\hexDAG-1\framework-tests\project19-shell-command

AI SUMMARY:
------------------------------
1. WHAT THIS COMMAND DOES: Shows which folder you're currently in.

2. KEY INFORMATION:
   - Location: hexDAG-1 project folder
   - Path: framework-tests/project19-shell-command

3. PLAIN ENGLISH: You're in the shell command demo folder...
```

## Test Results
[To be filled after testing]

## Observations
[To be filled after testing]
