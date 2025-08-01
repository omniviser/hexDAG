#!/usr/bin/env python3
"""
🎓 hexAI Examples Runner

Run all examples or specific categories to learn hexAI progressively.

Usage:
    python examples/run_all.py                    # Run all examples
    python examples/run_all.py --category=basic   # Run basic examples only
    python examples/run_all.py --example=01       # Run specific example
    python examples/run_all.py --list             # List all examples
"""

import os
from pathlib import Path
import subprocess  # nosec B404 - trusted example runner only
import sys
import time
from typing import Any

# Example categories for organized learning
EXAMPLE_CATEGORIES = {
    "basic": ["01_basic_dag.py", "02_dependencies.py", "03_validation_basics.py"],
    "core": [
        "04_validation_strategies.py",
        "05_event_system.py",
        "06_ports_and_adapters.py",
        "07_error_handling.py",
    ],
    "nodes": ["08_function_nodes.py", "09_llm_nodes.py", "10_agent_nodes.py"],
    "visualization": ["11_dag_visualization.py", "12_data_aggregation.py"],
    "enterprise": ["13_yaml_pipelines.py", "14_pipeline_compilation.py", "15_pipeline_catalog.py"],
    "problems": [
        "16_validation_errors.py",
        "17_performance_optimization.py",
        "18_advanced_patterns.py",
    ],
    "advanced": ["19_complex_workflow.py", "20_integration_testing.py"],
}

# All examples in learning order
ALL_EXAMPLES = [
    "01_basic_dag.py",
    "02_dependencies.py",
    "03_validation_basics.py",
    "04_validation_strategies.py",
    "05_event_system.py",
    "06_ports_and_adapters.py",
    "07_error_handling.py",
    "08_function_nodes.py",
    "09_llm_nodes.py",
    "10_agent_nodes.py",
    "11_dag_visualization.py",
    "12_data_aggregation.py",
    "13_yaml_pipelines.py",
    "14_pipeline_compilation.py",
    "15_pipeline_catalog.py",
    "16_validation_errors.py",
    "17_performance_optimization.py",
    "18_advanced_patterns.py",
    "19_complex_workflow.py",
    "20_integration_testing.py",
]


def list_examples():
    """List all available examples."""
    print("📚 Available hexAI Examples")
    print("=" * 40)

    for category, examples in EXAMPLE_CATEGORIES.items():
        print(f"\n🔧 {category.upper()} Examples:")
        for example in examples:
            print(f"   • {example}")

    print(f"\n📊 Total: {len(ALL_EXAMPLES)} examples across {len(EXAMPLE_CATEGORIES)} categories")


def run_example(example_path: str, quiet: bool = False) -> bool:
    """Run a single example and return success status."""
    try:
        print(f"\n🚀 Running {example_path}...")
        print("-" * 50)

        # Run the example with PYTHONPATH set from fastapi_app directory
        env = os.environ.copy()
        env["PYTHONPATH"] = "src"

        result = subprocess.run(  # nosec B603 - trusted example execution only
            [sys.executable, f"examples/{example_path}"],
            cwd=Path(__file__).parent.parent,  # Run from fastapi_app directory
            capture_output=quiet,  # Capture output if quiet mode
            text=True,
            timeout=60,  # 60 second timeout
            env=env,
        )

        if result.returncode == 0:
            print(f"✅ {example_path} completed successfully")
            return True
        else:
            print(f"❌ {example_path} failed with return code {result.returncode}")
            if quiet and result.stderr:
                print(f"Error: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ {example_path} timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"💥 {example_path} failed with exception: {e}")
        return False


def run_category(category: str, quiet: bool = False) -> dict[str, Any]:
    """Run all examples in a category."""
    if category not in EXAMPLE_CATEGORIES:
        print(f"❌ Unknown category: {category}")
        print(f"Available categories: {list(EXAMPLE_CATEGORIES.keys())}")
        return {"success": False, "results": {}}

    examples = EXAMPLE_CATEGORIES[category]
    print(f"\n🎯 Running {category.upper()} examples ({len(examples)} examples)")
    print("=" * 60)

    results = {}
    start_time = time.time()

    for example in examples:
        success = run_example(example, quiet)
        results[example] = success

        if not success:
            print(f"⚠️  Skipping remaining examples in category due to failure")
            break

    end_time = time.time()
    total_time = end_time - start_time

    # Summary
    successful = sum(1 for success in results.values() if success)
    total = len(results)

    print(f"\n📊 {category.upper()} Category Summary:")
    print(f"   • Successful: {successful}/{total}")
    print(f"   • Total time: {total_time:.2f}s")
    print(f"   • Average time per example: {total_time/max(1, total):.2f}s")

    return {
        "success": successful == total,
        "results": results,
        "total_time": total_time,
        "successful": successful,
        "total": total,
    }


def run_all_examples(quiet: bool = False) -> dict[str, Any]:
    """Run all examples in order."""
    print(f"\n🎓 Running ALL hexAI Examples ({len(ALL_EXAMPLES)} examples)")
    print("=" * 60)

    results = {}
    start_time = time.time()

    for example in ALL_EXAMPLES:
        success = run_example(example, quiet)
        results[example] = success

        if not success:
            print(f"⚠️  Example {example} failed, but continuing with remaining examples")

    end_time = time.time()
    total_time = end_time - start_time

    # Summary
    successful = sum(1 for success in results.values() if success)
    total = len(results)

    print(f"\n📊 Overall Summary:")
    print(f"   • Successful: {successful}/{total}")
    print(f"   • Total time: {total_time:.2f}s")
    print(f"   • Average time per example: {total_time/max(1, total):.2f}s")

    return {
        "success": successful == total,
        "results": results,
        "total_time": total_time,
        "successful": successful,
        "total": total,
    }


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run hexAI learning examples")
    parser.add_argument("--category", help="Run examples from specific category")
    parser.add_argument("--example", help="Run specific example (e.g., 01_basic_dag.py)")
    parser.add_argument("--list", action="store_true", help="List all available examples")
    parser.add_argument("--all", action="store_true", help="Run all examples")
    parser.add_argument("--quiet", action="store_true", help="Run examples without showing output")

    args = parser.parse_args()

    # Check if we're in the right directory
    examples_dir = Path(__file__).parent
    if not examples_dir.exists():
        print("❌ Examples directory not found. Please run from fastapi_app directory.")
        sys.exit(1)

    if args.list:
        list_examples()
        return

    if args.example:
        # Run specific example
        example_path = args.example
        if not example_path.endswith(".py"):
            example_path = f"{example_path}.py"

        if not (examples_dir / example_path).exists():
            print(f"❌ Example {example_path} not found")
            print(f"Available examples: {ALL_EXAMPLES}")
            return

        success = run_example(example_path, args.quiet)
        sys.exit(0 if success else 1)

    elif args.category:
        # Run category
        result = run_category(args.category, args.quiet)
        sys.exit(0 if result["success"] else 1)

    elif args.all:
        # Run all examples
        result = run_all_examples(args.quiet)
        sys.exit(0 if result["success"] else 1)

    else:
        # Default: show help
        print("🎓 hexAI Examples Runner")
        print("=" * 30)
        print("\nUsage:")
        print("  python examples/run_all.py --list                    # List all examples")
        print("  python examples/run_all.py --example=01_basic_dag    # Run specific example")
        print("  python examples/run_all.py --category=basic          # Run category")
        print("  python examples/run_all.py --all                     # Run all examples")
        print(
            "  python examples/run_all.py --quiet                   # Run without showing example output"
        )
        print("\nExamples:")
        print(
            "  python examples/run_all.py --category=basic --quiet  # Run basic examples silently"
        )
        print("  python examples/run_all.py --all --quiet             # Run all examples silently")
        print(
            "  python examples/run_all.py --example=05_event_system # Run specific example with output"
        )
        print("\nCategories:")
        for category in EXAMPLE_CATEGORIES.keys():
            print(f"  • {category}: {len(EXAMPLE_CATEGORIES[category])} examples")

        print(f"\nTotal: {len(ALL_EXAMPLES)} examples available")


if __name__ == "__main__":
    main()
