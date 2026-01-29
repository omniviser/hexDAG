#!/usr/bin/env python3
"""
Pre-commit hook to verify all Python examples run successfully.

This script ensures that:
1. All Python examples in the examples/ directory execute without errors
2. No examples are broken due to recent code changes

Note: Notebooks are validated separately by check_notebooks.py
"""

import subprocess  # nosec B404 # Needed to run example validation
import sys
from pathlib import Path


def run_python_examples() -> tuple[bool, str]:
    """Run all Python example files.

    Returns
    -------
        Tuple of (success, output_message)
    """
    examples_dir = Path("examples")

    if not examples_dir.exists():
        return False, "Examples directory not found"

    # Find all Python files in examples directory (excluding __pycache__, etc.)
    python_examples = sorted(examples_dir.glob("**/*.py"))
    python_examples = [
        p for p in python_examples if "__pycache__" not in str(p) and "plugins" not in str(p)
    ]

    if not python_examples:
        return True, "No Python examples found (all converted to notebooks)"

    failed_examples = []
    passed_count = 0

    print(f"   Found {len(python_examples)} Python example(s) to check")

    for example_file in python_examples:
        relative_path = example_file.relative_to(examples_dir)
        print(f"   Running {relative_path}... ", end="", flush=True)

        try:
            result = subprocess.run(  # nosec B603 B607 # Safe: controlled command
                ["uv", "run", "python", str(example_file)],
                capture_output=True,
                text=True,
                timeout=60,  # 1 minute per example
            )

            if result.returncode == 0:
                print("✅")
                passed_count += 1
            else:
                print("❌")
                error_msg = result.stderr[:200] if result.stderr else result.stdout[:200]
                failed_examples.append((relative_path, error_msg))

        except subprocess.TimeoutExpired:
            print("⏱️ TIMEOUT")
            failed_examples.append((relative_path, "Execution timed out after 60 seconds"))
        except Exception as e:
            print("💥 ERROR")
            failed_examples.append((relative_path, str(e)))

    if failed_examples:
        error_details = []
        for example, error in failed_examples:
            error_details.append(f"   • {example}: {error}")

        return False, (
            f"{passed_count}/{len(python_examples)} examples passed\n"
            "Failed examples:\n" + "\n".join(error_details)
        )

    return True, f"All {passed_count} Python examples passed!"


def main() -> int:
    """Check examples and return exit code.

    Returns
    -------
        Exit code (0 for success, 1 for failure).
    """
    print("🎓 Checking Python examples functionality...")

    success, message = run_python_examples()

    if success:
        print(f"✅ {message}")
        return 0

    print("❌ Examples check failed!")
    print(message)
    print()
    print("💡 To fix this:")
    print("   1. Run failing examples manually to see detailed errors:")
    print("      uv run python examples/<failing_example>.py")
    print("   2. Fix any broken examples before committing")
    print("   3. Consider converting complex examples to notebooks in notebooks/")
    print()
    print("🚫 Commit blocked - examples must pass before committing changes")
    return 1


if __name__ == "__main__":
    sys.exit(main())
