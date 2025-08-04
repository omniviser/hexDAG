#!/usr/bin/env python3
"""
Pre-commit hook to verify all examples run successfully.

This script ensures that:
1. All examples in the examples/ directory execute without errors
2. The example runner (run_all.py) completes successfully
3. No examples are broken due to recent code changes

This prevents commits that would break the examples, ensuring they remain
functional for users learning the library.
"""

import subprocess  # nosec B404 # Needed to run example validation
import sys
from pathlib import Path


def run_examples() -> tuple[bool, str]:
    """Run all examples using the run_all.py script.

    Returns
    -------
        Tuple of (success, output_message)
    """
    examples_dir = Path("examples")
    run_all_script = examples_dir / "run_all.py"

    if not examples_dir.exists():
        return False, "Examples directory not found"

    if not run_all_script.exists():
        return False, "run_all.py script not found in examples directory"

    try:
        # Run the examples using poetry to ensure correct environment
        result = subprocess.run(  # nosec B603 B607 # Safe: controlled command, fixed args
            ["poetry", "run", "python", "run_all.py", "--all", "--quiet"],
            cwd=examples_dir,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout for all examples
        )

        if result.returncode == 0:
            # Parse the output to get success count
            output_lines = result.stdout.strip().split("\n")
            summary_line = None
            for line in reversed(output_lines):
                if "Successful:" in line:
                    summary_line = line.strip()
                    break

            if summary_line:
                return True, f"All examples passed! {summary_line}"
            else:
                return True, "All examples completed successfully"
        else:
            # Extract error information
            error_info = []
            in_failed_section = False

            for line in result.stdout.split("\n"):
                if "failed with return code" in line.lower():
                    in_failed_section = True
                    error_info.append(line.strip())
                elif in_failed_section and line.strip():
                    error_info.append(line.strip())
                elif "Overall Summary:" in line:
                    in_failed_section = False

            if error_info:
                return False, "Examples failed:\n" + "\n".join(error_info[:10])  # Limit output
            else:
                stderr_excerpt = result.stderr[:500]
                return False, (
                    f"Examples failed with return code {result.returncode}\n"
                    f"Stderr: {stderr_excerpt}"
                )

    except subprocess.TimeoutExpired:
        return False, "Examples timed out after 5 minutes"
    except subprocess.CalledProcessError as e:
        return False, f"Failed to run examples: {e}"
    except Exception as e:
        return False, f"Unexpected error: {e}"


def main() -> int:
    """Check examples and return exit code."""
    print("🎓 Checking examples functionality...")

    success, message = run_examples()

    if success:
        print(f"✅ {message}")
        return 0
    else:
        print("❌ Examples check failed!")
        print(f"   {message}")
        print()
        print("💡 To fix this:")
        print("   1. Check the failing examples manually:")
        print("      cd examples && poetry run python run_all.py --all")
        print("   2. Fix any broken examples before committing")
        print("   3. Ensure all dependencies are properly installed")
        print()
        print("🚫 Commit blocked - examples must pass before committing changes")
        return 1


if __name__ == "__main__":
    sys.exit(main())
