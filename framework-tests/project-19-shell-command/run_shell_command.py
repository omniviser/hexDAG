#!/usr/bin/env python
"""
hexDAG Shell Command Execution and Summarization Demo
Ported from LangGraph project-19-shell-command-execution-and-summarization

Pattern: Command Execution + LLM Summary
- Execute shell commands safely
- Summarize output in human-readable format
- Explain what the command does

SECURITY NOTE: Only predefined safe commands are allowed.
Never execute arbitrary user input without validation!

Run with: ..\..\.venv\Scripts\python.exe run_shell_command.py
"""
import asyncio
import os
import subprocess
import sys
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


# Safe commands whitelist (read-only, no destructive operations)
# Windows and Unix compatible
SAFE_COMMANDS = {
    "list_files": {
        "windows": "dir",
        "unix": "ls -la",
        "description": "List files in current directory"
    },
    "disk_space": {
        "windows": "wmic logicaldisk get size,freespace,caption",
        "unix": "df -h",
        "description": "Show disk space usage"
    },
    "system_info": {
        "windows": "systeminfo | findstr /B /C:\"OS Name\" /C:\"OS Version\" /C:\"System Type\" /C:\"Total Physical Memory\"",
        "unix": "uname -a",
        "description": "Show system information"
    },
    "current_directory": {
        "windows": "cd",
        "unix": "pwd",
        "description": "Show current working directory"
    },
    "environment": {
        "windows": "set",
        "unix": "env",
        "description": "Show environment variables"
    },
    "network_info": {
        "windows": "ipconfig",
        "unix": "ifconfig 2>/dev/null || ip addr",
        "description": "Show network configuration"
    },
    "running_processes": {
        "windows": "tasklist /FI \"STATUS eq RUNNING\" | more",
        "unix": "ps aux | head -20",
        "description": "Show running processes"
    },
    "python_version": {
        "windows": "python --version",
        "unix": "python3 --version",
        "description": "Show Python version"
    },
}


def get_platform():
    """Detect operating system."""
    return "windows" if sys.platform == "win32" else "unix"


def run_command_safe(command_key: str, timeout: int = 30) -> dict:
    """
    Execute a predefined safe command.

    LangGraph version:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

    hexDAG version: Only allows whitelisted commands with timeout.
    """
    if command_key not in SAFE_COMMANDS:
        return {
            "success": False,
            "output": f"Error: Unknown command '{command_key}'",
            "command": command_key
        }

    platform = get_platform()
    command_info = SAFE_COMMANDS[command_key]
    command = command_info[platform]
    description = command_info["description"]

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if result.returncode == 0:
            output = result.stdout.strip()
            success = True
        else:
            output = result.stderr.strip() or result.stdout.strip() or "Command failed with no output"
            success = False

        return {
            "success": success,
            "output": output,
            "command": command,
            "description": description
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "output": f"Error: Command timed out after {timeout} seconds",
            "command": command,
            "description": description
        }
    except Exception as e:
        return {
            "success": False,
            "output": f"Error: {str(e)}",
            "command": command,
            "description": description
        }


async def execute_command(inputs: dict) -> dict:
    """
    Execute a shell command safely.

    Only predefined safe commands are allowed.
    """
    command_key = inputs.get("command_key", "list_files")

    print(f"  [EXECUTE] Running: {command_key}")

    result = run_command_safe(command_key)

    if result["success"]:
        print(f"  [EXECUTE] Command completed successfully")
    else:
        print(f"  [EXECUTE] Command failed or returned error")

    return {
        "command": result["command"],
        "command_output": result["output"],
        "description": result.get("description", ""),
        "success": result["success"]
    }


async def summarize_output(inputs: dict) -> dict:
    """
    Summarize command output in human-readable format.

    LangGraph version:
        response = llm.invoke(f"Summarize the following command output: {output}")

    hexDAG version: More detailed explanation with context.
    """
    command = inputs.get("command", "")
    command_output = inputs.get("command_output", "")
    description = inputs.get("description", "")
    success = inputs.get("success", True)

    print(f"  [SUMMARIZE] Analyzing command output...")

    status = "completed successfully" if success else "encountered an error"

    prompt = f"""You are a helpful system administrator assistant. Explain this command output to a non-technical user.

COMMAND: {command}
PURPOSE: {description}
STATUS: {status}

OUTPUT:
{command_output[:3000]}  # Truncate very long output

Provide:
1. WHAT THIS COMMAND DOES (1 sentence)
2. KEY INFORMATION (bullet points of important findings)
3. PLAIN ENGLISH SUMMARY (explain like I'm not technical)
4. ANY CONCERNS (warnings or issues noticed, or "None" if all looks good)

Keep it concise and helpful."""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    summary = response.text.strip()
    print(f"  [SUMMARIZE] Summary generated")

    return {
        "summary": summary,
        "command": command
    }


async def run_shell_demo():
    """
    Demonstrate shell command execution and summarization.
    """
    print("=" * 60)
    print("hexDAG Shell Command Agent Demo")
    print("=" * 60)
    print(f"Platform: {get_platform()}")
    print()

    # Commands to demonstrate
    demo_commands = [
        "current_directory",
        "list_files",
        "python_version",
        "disk_space",
    ]

    for i, cmd_key in enumerate(demo_commands, 1):
        cmd_info = SAFE_COMMANDS.get(cmd_key, {})
        print(f"[Command {i}] {cmd_key}: {cmd_info.get('description', '')}")
        print("-" * 50)

        # Execute command
        exec_result = await execute_command({"command_key": cmd_key})

        # Summarize output
        summary_input = {
            "command": exec_result["command"],
            "command_output": exec_result["command_output"],
            "description": exec_result["description"],
            "success": exec_result["success"]
        }
        summary_result = await summarize_output(summary_input)

        # Display
        print()
        print("RAW OUTPUT:")
        print("-" * 30)
        # Truncate long output for display
        output = exec_result["command_output"]
        if len(output) > 500:
            output = output[:500] + "\n... (truncated)"
        print(output or "(no output)")
        print()
        print("AI SUMMARY:")
        print("-" * 30)
        print(summary_result["summary"])
        print()
        print("=" * 60)
        print()


async def main():
    """Main entry point."""
    print("Available commands:")
    for key, info in SAFE_COMMANDS.items():
        print(f"  - {key}: {info['description']}")
    print()

    await run_shell_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
