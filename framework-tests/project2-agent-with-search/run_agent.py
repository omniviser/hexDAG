#!/usr/bin/env python
"""
hexDAG Agent with Search Tool
Ported from LangGraph project-02-agent-with-search-tool
Uses Google Gemini API + DuckDuckGo Search (free, no API key needed)

Run with: ..\..\.venv\Scripts\python.exe run_agent.py
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
    print("WARNING: duckduckgo-search not installed. Install with: pip install duckduckgo-search")
    HAS_DDGS = False


def web_search(query: str, max_results: int = 3) -> str:
    """
    Search the web using DuckDuckGo (free, no API key needed).
    Returns formatted search results.
    """
    if not HAS_DDGS:
        return f"[Search unavailable] Query was: {query}"

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return f"No results found for: {query}"

        # Format results
        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get('title', 'No title')
            body = r.get('body', 'No description')
            href = r.get('href', '')
            formatted.append(f"{i}. {title}\n   {body}\n   URL: {href}")

        return "\n\n".join(formatted)
    except Exception as e:
        return f"Search error: {str(e)}"


async def agent_node(inputs: dict) -> dict:
    """
    Agent node that decides whether to search or respond.
    Uses ReAct-style reasoning.
    """
    messages = inputs.get("messages", [])
    user_input = inputs.get("user_input", "")
    search_results = inputs.get("search_results", "")

    model = genai.GenerativeModel("gemini-2.0-flash")

    # Build prompt
    system_prompt = """You are a research assistant with access to web search.

IMPORTANT RULES:
1. If the user asks a factual question, you MUST search the web first
2. After seeing search results, provide a helpful answer based on them
3. Always cite your sources when using search results
4. If you have search results, use them to answer

Available action:
- To search: respond with exactly "SEARCH: <your search query>"
- To answer: just provide your answer directly"""

    prompt_parts = [system_prompt, ""]

    # Add conversation history
    if messages:
        prompt_parts.append("Conversation history:")
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"{role}: {content}")
        prompt_parts.append("")

    # Add search results if available
    if search_results:
        prompt_parts.append(f"Search results:\n{search_results}")
        prompt_parts.append("")
        prompt_parts.append("Now provide a helpful answer based on these search results.")
    else:
        prompt_parts.append(f"User question: {user_input}")
        prompt_parts.append("")
        prompt_parts.append("Decide: should you search the web first, or can you answer directly?")

    full_prompt = "\n".join(prompt_parts)

    # Generate response
    response = model.generate_content(full_prompt)
    response_text = response.text.strip()

    # Check if agent wants to search
    needs_search = False
    search_query = ""

    if response_text.upper().startswith("SEARCH:"):
        needs_search = True
        search_query = response_text[7:].strip()

    return {
        "response": response_text,
        "needs_search": needs_search,
        "search_query": search_query,
    }


async def search_node(inputs: dict) -> dict:
    """
    Search node that performs web search.
    """
    search_query = inputs.get("search_query", "")

    if not search_query:
        return {"search_results": ""}

    print(f"  [Searching web for: {search_query}]")
    results = web_search(search_query)

    return {"search_results": results}


async def run_agent_loop(user_input: str, messages: list) -> tuple[str, list]:
    """
    Run the agent loop: agent -> (optional search) -> agent -> response
    """
    # Create graph with agent and search nodes
    graph = DirectedGraph()
    graph.add(NodeSpec("agent", agent_node))

    orchestrator = Orchestrator()

    # First agent call - decide if search is needed
    result = await orchestrator.run(graph, {
        "user_input": user_input,
        "messages": messages,
        "search_results": ""
    })

    agent_output = result.get("agent", {})
    needs_search = agent_output.get("needs_search", False)

    if needs_search:
        # Perform search
        search_query = agent_output.get("search_query", "")
        search_results = web_search(search_query)

        # Second agent call with search results
        result = await orchestrator.run(graph, {
            "user_input": user_input,
            "messages": messages,
            "search_results": search_results
        })
        agent_output = result.get("agent", {})

    response = agent_output.get("response", "No response")

    # Clean up SEARCH: prefix if still present
    if response.upper().startswith("SEARCH:"):
        response = "I tried to search but encountered an issue. Please try again."

    # Update messages
    messages.append({"role": "user", "content": user_input})
    messages.append({"role": "assistant", "content": response})

    return response, messages


async def main():
    """Main interaction loop."""
    print("=" * 60)
    print("hexDAG Research Agent (Ported from LangGraph)")
    print("Using Google Gemini + DuckDuckGo Search")
    print("=" * 60)
    print("Type 'quit' to exit.\n")

    if not HAS_DDGS:
        print("NOTE: Install duckduckgo-search for web search capability")
        print("      pip install duckduckgo-search\n")

    conversation_messages = []

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

        response, conversation_messages = await run_agent_loop(
            user_input,
            conversation_messages
        )

        print(f"Agent: {response}\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
