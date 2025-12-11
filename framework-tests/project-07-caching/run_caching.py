#!/usr/bin/env python
"""
hexDAG Node-Level Caching Demo
Ported from LangGraph project-07-document-summarization-pipeline

Pattern: Cache LLM responses to avoid redundant API calls
- First call: Makes LLM request, stores in cache
- Second call (same input): Returns cached result (no API call)

Run with: ..\..\.venv\Scripts\python.exe run_caching.py
"""
import asyncio
import hashlib
import os
import sys
import time
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


# Simple in-memory cache (equivalent to LangGraph's SqliteSaver with :memory:)
class SimpleCache:
    """
    Simple cache implementation.

    LangGraph uses SqliteSaver for this:
        memory = SqliteSaver.from_conn_string(":memory:")
        app = graph.compile(checkpointer=memory)

    hexDAG doesn't have built-in checkpointer, so we implement manually.
    """

    def __init__(self):
        self._cache = {}
        self._hits = 0
        self._misses = 0

    def _hash_key(self, key: str) -> str:
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, key: str):
        hashed = self._hash_key(key)
        if hashed in self._cache:
            self._hits += 1
            return self._cache[hashed]
        self._misses += 1
        return None

    def set(self, key: str, value):
        hashed = self._hash_key(key)
        self._cache[hashed] = value

    def stats(self):
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total_entries": len(self._cache)
        }


# Global cache instance
cache = SimpleCache()


async def chatbot_with_cache(inputs: dict) -> dict:
    """
    Chatbot node with caching.

    If the same question was asked before, return cached response.
    Otherwise, call LLM and cache the result.
    """
    user_input = inputs.get("user_input", "")
    thread_id = inputs.get("thread_id", "default")

    # Create cache key from thread_id + user_input (like LangGraph's config)
    cache_key = f"{thread_id}:{user_input}"

    # Check cache first
    cached_response = cache.get(cache_key)
    if cached_response:
        print(f"  [CACHE HIT] Returning cached response")
        return {
            "response": cached_response,
            "from_cache": True
        }

    # Cache miss - call LLM
    print(f"  [CACHE MISS] Calling LLM...")
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(user_input)
    response_text = response.text.strip()

    # Store in cache
    cache.set(cache_key, response_text)

    return {
        "response": response_text,
        "from_cache": False
    }


async def run_caching_demo():
    """
    Demonstrate caching behavior.

    Same as LangGraph example:
    1. Ask "What is the capital of France?" - calls LLM
    2. Ask same question again - returns cached result (no LLM call)
    """
    print("=" * 60)
    print("hexDAG Node-Level Caching Demo")
    print("=" * 60)
    print()

    # Create graph
    graph = DirectedGraph()
    graph.add(NodeSpec("chatbot", chatbot_with_cache))

    orchestrator = Orchestrator()
    thread_id = "1"  # Same as LangGraph config

    # Test question (same as LangGraph example)
    question = "What is the capital of France?"

    # --- First call (should be cache MISS) ---
    print(f"[Call 1] Question: {question}")
    start = time.time()

    result1 = await orchestrator.run(graph, {
        "user_input": question,
        "thread_id": thread_id
    })

    time1 = (time.time() - start) * 1000
    output1 = result1.get("chatbot", {})
    print(f"Response: {output1.get('response', 'No response')[:100]}...")
    print(f"From cache: {output1.get('from_cache', False)}")
    print(f"Time: {time1:.2f}ms")
    print()

    # --- Second call (should be cache HIT) ---
    print(f"[Call 2] Question: {question} (same question)")
    start = time.time()

    result2 = await orchestrator.run(graph, {
        "user_input": question,
        "thread_id": thread_id
    })

    time2 = (time.time() - start) * 1000
    output2 = result2.get("chatbot", {})
    print(f"Response: {output2.get('response', 'No response')[:100]}...")
    print(f"From cache: {output2.get('from_cache', False)}")
    print(f"Time: {time2:.2f}ms")
    print()

    # --- Third call with different question ---
    question2 = "What is the capital of Germany?"
    print(f"[Call 3] Question: {question2} (different question)")
    start = time.time()

    result3 = await orchestrator.run(graph, {
        "user_input": question2,
        "thread_id": thread_id
    })

    time3 = (time.time() - start) * 1000
    output3 = result3.get("chatbot", {})
    print(f"Response: {output3.get('response', 'No response')[:100]}...")
    print(f"From cache: {output3.get('from_cache', False)}")
    print(f"Time: {time3:.2f}ms")
    print()

    # --- Summary ---
    print("=" * 60)
    print("CACHE STATISTICS")
    print("=" * 60)
    stats = cache.stats()
    print(f"Cache hits: {stats['hits']}")
    print(f"Cache misses: {stats['misses']}")
    print(f"Total cached entries: {stats['total_entries']}")
    print()
    print(f"Time saved on cached call: {time1 - time2:.2f}ms")


async def main():
    """Main entry point."""
    await run_caching_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
