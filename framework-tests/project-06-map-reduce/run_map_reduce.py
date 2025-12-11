#!/usr/bin/env python
"""
hexDAG Map-Reduce Workflow
Ported from LangGraph project-06-iterative-processing-workflow

Pattern: Process a list of items, then aggregate results
- MAP: Process each item with LLM (can run in parallel)
- REDUCE: Aggregate all results into summary

Run with: ..\..\.venv\Scripts\python.exe run_map_reduce.py
"""
import asyncio
import os
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


async def process_single_item(item: str) -> dict:
    """
    Process a single item with LLM.
    Equivalent to LangGraph's process_item function.
    """
    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = f"Process this item and describe it in one sentence: {item}"
    response = model.generate_content(prompt)

    return {
        "item": item,
        "result": response.text.strip()
    }


async def process_items_node(inputs: dict) -> dict:
    """
    MAP phase: Process all items.
    In hexDAG, we can run these in parallel using asyncio.gather.
    """
    items = inputs.get("items", [])

    print(f"  [MAP] Processing {len(items)} items...")

    # Process all items in parallel (this is the "map" part)
    tasks = [process_single_item(item) for item in items]
    results = await asyncio.gather(*tasks)

    print(f"  [MAP] Completed processing {len(results)} items")

    return {
        "processed_results": list(results)
    }


async def aggregate_node(inputs: dict) -> dict:
    """
    REDUCE phase: Aggregate all processed results.
    Equivalent to LangGraph's aggregate_results function.
    """
    processed_results = inputs.get("processed_results", [])

    print(f"  [REDUCE] Aggregating {len(processed_results)} results...")

    # Build summary
    summary_lines = []
    for r in processed_results:
        item = r.get("item", "unknown")
        result = r.get("result", "no result")
        summary_lines.append(f"- {item}: {result}")

    summary = "\n".join(summary_lines)
    total = len(processed_results)

    return {
        "summary": summary,
        "total_items": total,
        "message": f"Aggregated results: {total} items processed"
    }


async def run_map_reduce(items: list[str]):
    """
    Run the Map-Reduce workflow.

    Graph structure:
        process_items (MAP) → aggregate (REDUCE)

    This is a clean DAG - hexDAG handles this pattern well!
    """
    print("=" * 60)
    print("hexDAG Map-Reduce Workflow")
    print("=" * 60)
    print(f"Items to process: {items}")
    print("=" * 60 + "\n")

    # Create DAG: process → aggregate
    graph = DirectedGraph()
    graph.add(NodeSpec("process_items", process_items_node))
    graph.add(NodeSpec("aggregate", aggregate_node).after("process_items"))

    orchestrator = Orchestrator()

    # Run the workflow
    print("[Starting Map-Reduce workflow]\n")
    result = await orchestrator.run(graph, {"items": items})

    # Extract results
    aggregate_output = result.get("aggregate", {})
    summary = aggregate_output.get("summary", "No summary")
    total = aggregate_output.get("total_items", 0)
    message = aggregate_output.get("message", "")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n{message}\n")
    print("Summary:")
    print(summary)
    print("\n" + "=" * 60)

    return result


async def main():
    """Main entry point."""
    # Default items (same as LangGraph example)
    default_items = ["apple", "banana", "cherry"]

    print("Enter items to process (comma-separated)")
    print(f"Or press Enter for default: {default_items}")
    user_input = input("> ").strip()

    if user_input:
        items = [item.strip() for item in user_input.split(",")]
    else:
        items = default_items

    await run_map_reduce(items)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
