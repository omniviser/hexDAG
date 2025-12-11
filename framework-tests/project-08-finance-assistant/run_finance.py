#!/usr/bin/env python
"""
hexDAG Finance Assistant Demo
Ported from LangGraph project-08-data-extraction-agent

Pattern: Agent with search tool for real-time financial data
- Search for stock prices and financial info
- LLM analyzes results and provides insights

Run with: ..\..\.venv\Scripts\python.exe run_finance.py
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
from duckduckgo_search import DDGS
from hexdag.core.domain.dag import DirectedGraph, NodeSpec
from hexdag.core.orchestration.orchestrator import Orchestrator

# Configure Google Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY not found in .env file")
    sys.exit(1)

genai.configure(api_key=GEMINI_API_KEY)


async def search_financial_data(inputs: dict) -> dict:
    """
    Search for financial/stock information using DuckDuckGo.

    LangGraph uses TavilySearchResults:
        tool = TavilySearchResults(max_results=2)
        tool.invoke(query)

    hexDAG uses DuckDuckGo (free, no API key):
        DDGS().text(query, max_results=5)
    """
    query = inputs.get("user_query", "")

    print(f"  [SEARCH] Searching for: {query}")

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        # Format results for LLM
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(
                f"{i}. {result.get('title', 'No title')}\n"
                f"   {result.get('body', 'No description')}\n"
                f"   Source: {result.get('href', 'No URL')}"
            )

        search_text = "\n\n".join(formatted_results) if formatted_results else "No results found"
        print(f"  [SEARCH] Found {len(results)} results")

        return {
            "search_results": search_text,
            "result_count": len(results)
        }
    except Exception as e:
        print(f"  [SEARCH] Error: {e}")
        return {
            "search_results": f"Search error: {str(e)}",
            "result_count": 0
        }


async def finance_analyst(inputs: dict) -> dict:
    """
    Analyze search results and provide financial insights.

    Takes search results and user query, returns helpful financial analysis.
    """
    user_query = inputs.get("user_query", "")
    search_results = inputs.get("search_results", "No search results")

    print(f"  [ANALYST] Analyzing results...")

    prompt = f"""You are a helpful finance assistant.

User question: {user_query}

Search results:
{search_results}

Based on the search results, provide a clear and helpful answer about the financial information requested.
If the search results don't contain the exact information, summarize what was found and note any limitations.
Be concise but informative."""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    return {
        "response": response.text.strip(),
        "query": user_query
    }


async def run_finance_demo():
    """
    Demonstrate finance assistant.

    Flow: User query -> Search -> LLM Analysis -> Response
    """
    print("=" * 60)
    print("hexDAG Finance Assistant Demo")
    print("=" * 60)
    print()

    # Create graph
    graph = DirectedGraph()
    graph.add(NodeSpec("search_tool", search_financial_data))
    graph.add(NodeSpec("finance_analyst", finance_analyst, depends_on=["search_tool"]))

    orchestrator = Orchestrator()

    # Test queries (similar to LangGraph example)
    queries = [
        "What is the current stock price of Apple AAPL?",
        "How is Microsoft MSFT performing today?",
        "What is the market cap of Tesla?",
    ]

    for i, query in enumerate(queries, 1):
        print(f"[Query {i}] {query}")
        print("-" * 50)

        result = await orchestrator.run(graph, {
            "user_query": query
        })

        analyst_output = result.get("finance_analyst", {})
        response = analyst_output.get("response", "No response")

        # Truncate long responses for display
        if len(response) > 500:
            response = response[:500] + "..."

        print(f"\nResponse:\n{response}")
        print()
        print("=" * 60)
        print()


async def interactive_mode():
    """
    Interactive finance assistant - ask your own questions.
    """
    print("=" * 60)
    print("hexDAG Finance Assistant - Interactive Mode")
    print("=" * 60)
    print("Ask questions about stocks, markets, and finance.")
    print("Type 'quit' to exit.")
    print()

    # Create graph
    graph = DirectedGraph()
    graph.add(NodeSpec("search_tool", search_financial_data))
    graph.add(NodeSpec("finance_analyst", finance_analyst, depends_on=["search_tool"]))

    orchestrator = Orchestrator()

    while True:
        try:
            query = input("You: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if not query:
                continue

            print()
            result = await orchestrator.run(graph, {
                "user_query": query
            })

            analyst_output = result.get("finance_analyst", {})
            response = analyst_output.get("response", "No response")

            print(f"\nAssistant: {response}")
            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break


async def main():
    """Main entry point."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        await interactive_mode()
    else:
        await run_finance_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
