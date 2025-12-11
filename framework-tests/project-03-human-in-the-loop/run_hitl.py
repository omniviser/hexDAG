#!/usr/bin/env python
"""
hexDAG Human-in-the-Loop Chatbot
Ported from LangGraph project-03-human-in-the-loop-chatbot

NOTE: hexDAG does not have native interrupt/resume like LangGraph.
This implementation uses a workaround with explicit approval steps.

Run with: ..\..\.venv\Scripts\python.exe run_hitl.py
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

# Try to import DuckDuckGo search
try:
    from duckduckgo_search import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False


def web_search(query: str, max_results: int = 3) -> str:
    """Search the web using DuckDuckGo."""
    if not HAS_DDGS:
        return f"[Search unavailable] Query was: {query}"

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return f"No results found for: {query}"

        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get('title', 'No title')
            body = r.get('body', 'No description')
            formatted.append(f"{i}. {title}: {body}")

        return "\n".join(formatted)
    except Exception as e:
        return f"Search error: {str(e)}"


async def search_node(inputs: dict) -> dict:
    """Node 1: Search the web for information."""
    question = inputs.get("question", "")
    print(f"  [Step 1: Searching web for: {question}]")

    results = web_search(question)
    return {"search_results": results}


async def draft_node(inputs: dict) -> dict:
    """Node 2: Draft an answer based on search results."""
    question = inputs.get("question", "")
    search_results = inputs.get("search_results", "")

    print(f"  [Step 2: Drafting answer...]")

    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = f"""Question: {question}

Found information:
{search_results}

Write a factual, helpful answer based on the above information.
Keep it concise but informative."""

    response = model.generate_content(prompt)
    draft = response.text.strip()

    return {"draft": draft}


def human_approval(draft: str) -> bool:
    """
    HUMAN-IN-THE-LOOP: Ask for human approval.

    This is the key difference from LangGraph:
    - LangGraph uses `interrupt()` to pause graph execution
    - hexDAG requires explicit function call (workaround)
    """
    print("\n" + "=" * 60)
    print("DRAFT ANSWER (awaiting approval):")
    print("-" * 60)
    print(draft)
    print("-" * 60)

    while True:
        response = input("Approve this answer? (yes/no/edit): ").strip().lower()

        if response in ['yes', 'y', 'approve']:
            return True, draft
        elif response in ['no', 'n', 'reject']:
            return False, draft
        elif response in ['edit', 'e']:
            print("Enter your edited version (end with empty line):")
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            edited = "\n".join(lines)
            return True, edited
        else:
            print("Please enter 'yes', 'no', or 'edit'")


async def finalize_node(inputs: dict) -> dict:
    """Node 3: Finalize the answer based on approval."""
    draft = inputs.get("draft", "")
    approved = inputs.get("approved", False)
    final_draft = inputs.get("final_draft", draft)

    if approved:
        return {"final_answer": final_draft}
    else:
        return {"final_answer": "Answer was not approved. Please try a different question."}


async def run_hitl_workflow(question: str) -> str:
    """
    Run the Human-in-the-Loop workflow.

    Workflow:
    1. Search web for information
    2. Draft answer based on search
    3. HUMAN APPROVAL (interrupt point)
    4. Finalize answer

    NOTE: In LangGraph, step 3 uses `interrupt()` to pause execution.
    In hexDAG, we implement this as explicit Python code between DAG runs.
    """

    # --- Phase 1: Search and Draft ---
    # Create graph for search + draft
    graph_phase1 = DirectedGraph()
    graph_phase1.add(NodeSpec("search", search_node))
    graph_phase1.add(NodeSpec("draft", draft_node).after("search"))

    orchestrator = Orchestrator()

    print("\n[Phase 1: Search and Draft]")
    result = await orchestrator.run(graph_phase1, {"question": question})

    draft = result.get("draft", {}).get("draft", "No draft generated")

    # --- Phase 2: Human Approval (THE INTERRUPT POINT) ---
    print("\n[Phase 2: Human Approval]")
    approved, final_draft = human_approval(draft)

    # --- Phase 3: Finalize ---
    print("\n[Phase 3: Finalizing]")
    graph_phase3 = DirectedGraph()
    graph_phase3.add(NodeSpec("finalize", finalize_node))

    result = await orchestrator.run(graph_phase3, {
        "draft": draft,
        "approved": approved,
        "final_draft": final_draft
    })

    final_answer = result.get("finalize", {}).get("final_answer", "No answer")
    return final_answer


async def main():
    """Main interaction loop."""
    print("=" * 60)
    print("hexDAG Human-in-the-Loop Agent")
    print("(Ported from LangGraph)")
    print("=" * 60)
    print()
    print("This agent will:")
    print("  1. Search the web for your question")
    print("  2. Draft an answer")
    print("  3. ASK FOR YOUR APPROVAL before finalizing")
    print()
    print("Type 'quit' to exit.\n")

    if not HAS_DDGS:
        print("NOTE: Install duckduckgo-search for web search")
        print("      pip install duckduckgo-search\n")

    while True:
        question = input("Your question: ")
        if question.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

        final_answer = await run_hitl_workflow(question)

        print("\n" + "=" * 60)
        print("FINAL ANSWER:")
        print("=" * 60)
        print(final_answer)
        print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
