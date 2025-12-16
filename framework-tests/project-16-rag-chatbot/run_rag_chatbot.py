#!/usr/bin/env python
"""
hexDAG RAG (Retrieval Augmented Generation) Chatbot Demo
Ported from LangGraph project-16-rag-retrieval-augmented-generation-chatbot

Pattern: RAG - Retrieve context then Generate answer
- Search knowledge base for relevant information
- Use LLM to answer questions based on retrieved context
- Reduces hallucination by grounding answers in facts

Run with: ..\..\.venv\Scripts\python.exe run_rag_chatbot.py
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


# Knowledge Base - Company Documentation
KNOWLEDGE_BASE = [
    {
        "id": "about",
        "title": "About hexDAG",
        "content": """
        hexDAG is an enterprise-ready AI agent orchestration framework that transforms
        complex AI workflows into deterministic, testable, and maintainable systems
        through declarative YAML configurations and DAG-based orchestration.

        Key features:
        - YAML-first declarative pipelines
        - Directed Acyclic Graph (DAG) based execution
        - Async-first architecture
        - Built-in validation with Pydantic
        - Hexagonal architecture for clean separation of concerns
        """
    },
    {
        "id": "langgraph",
        "title": "About LangGraph",
        "content": """
        LangGraph is a library for building stateful, multi-actor applications with LLMs.
        It is built on top of LangChain. LangGraph allows you to define workflows as
        graphs where nodes are functions and edges define the flow.

        Key features:
        - StateGraph for managing conversation state
        - Support for cycles (loops) in workflows
        - Built-in checkpointing with SqliteSaver
        - Human-in-the-loop interrupts
        """
    },
    {
        "id": "comparison",
        "title": "hexDAG vs LangGraph",
        "content": """
        Comparison of hexDAG and LangGraph:

        hexDAG:
        - DAG-based (no cycles) - always completes
        - YAML-first configuration
        - Predictable execution order
        - Enterprise-focused with strong typing

        LangGraph:
        - Supports cycles (iterate until done)
        - Python-first with StateGraph
        - Built-in human-in-the-loop
        - Built-in checkpointing/caching
        """
    },
    {
        "id": "rag",
        "title": "What is RAG?",
        "content": """
        RAG (Retrieval Augmented Generation) is a pattern that combines:
        1. Retrieval: Search a knowledge base for relevant information
        2. Augmentation: Add retrieved context to the prompt
        3. Generation: LLM generates answer using the context

        Benefits:
        - Reduces hallucination
        - Provides up-to-date information
        - Enables domain-specific knowledge
        - More accurate and grounded answers
        """
    },
    {
        "id": "pricing",
        "title": "Pricing Information",
        "content": """
        hexDAG Pricing:
        - Open Source: Free for all features
        - Enterprise Support: Contact sales for pricing
        - Cloud Hosting: Coming soon

        LangGraph/LangChain Pricing:
        - LangChain: Open source, free
        - LangSmith: Paid observability platform
        - LangServe: Free deployment tools
        """
    }
]


def simple_search(query: str, knowledge_base: list, top_k: int = 2) -> list:
    """
    Simple keyword-based search over knowledge base.

    In production, you'd use:
    - FAISS (like LangGraph example)
    - Pinecone, Weaviate, Chroma
    - Elasticsearch
    """
    query_words = set(query.lower().split())

    scored_docs = []
    for doc in knowledge_base:
        doc_text = (doc["title"] + " " + doc["content"]).lower()
        doc_words = set(doc_text.split())

        # Simple relevance score
        score = len(query_words.intersection(doc_words))

        # Boost for title matches
        title_words = set(doc["title"].lower().split())
        if query_words.intersection(title_words):
            score += 5

        scored_docs.append((score, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_docs[:top_k] if score > 0]


async def retrieve_context(inputs: dict) -> dict:
    """
    Retrieve relevant context from knowledge base.

    LangGraph version:
        docs = retriever.get_relevant_documents(query)
        context = "\n".join([d.page_content for d in docs])

    hexDAG version: Simple keyword search (could use FAISS too).
    """
    question = inputs.get("question", "")

    print(f"  [RETRIEVE] Searching knowledge base...")

    relevant_docs = simple_search(question, KNOWLEDGE_BASE, top_k=2)

    if relevant_docs:
        context_parts = []
        for doc in relevant_docs:
            context_parts.append(f"### {doc['title']}\n{doc['content']}")

        context = "\n\n".join(context_parts)
        print(f"  [RETRIEVE] Found {len(relevant_docs)} relevant documents")
    else:
        context = "No relevant information found in the knowledge base."
        print(f"  [RETRIEVE] No matching documents found")

    return {
        "context": context,
        "num_docs": len(relevant_docs)
    }


async def generate_answer(inputs: dict) -> dict:
    """
    Generate answer using retrieved context.

    LangGraph version:
        response = llm.invoke(f"Answer based on context: {context}")

    hexDAG version: Same pattern with structured prompt.
    """
    context = inputs.get("context", "")
    question = inputs.get("question", "")

    print(f"  [GENERATE] Generating answer...")

    prompt = f"""Answer the question based ONLY on the following context.
If the answer is not in the context, say "I don't have information about that in my knowledge base."

CONTEXT:
{context}

QUESTION: {question}

Provide a clear, helpful answer based on the context above."""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    answer = response.text.strip()
    print(f"  [GENERATE] Answer generated")

    return {
        "answer": answer,
        "question": question
    }


async def run_rag_demo():
    """
    Demonstrate RAG chatbot with various questions.
    """
    print("=" * 60)
    print("hexDAG RAG Chatbot Demo")
    print("=" * 60)
    print()

    # Test questions
    questions = [
        "What is LangGraph?",
        "What is hexDAG?",
        "What is RAG and why is it useful?",
        "How does hexDAG compare to LangGraph?",
        "What is the pricing for hexDAG?",
        "What is the weather today?",  # Not in knowledge base
    ]

    for i, question in enumerate(questions, 1):
        print(f"[Q{i}] {question}")
        print("-" * 50)

        # Retrieve context
        retrieve_result = await retrieve_context({"question": question})

        # Generate answer
        generate_input = {
            "question": question,
            "context": retrieve_result["context"]
        }
        generate_result = await generate_answer(generate_input)

        print(f"\nAnswer: {generate_result['answer']}")
        print()
        print("=" * 60)
        print()


async def interactive_mode():
    """
    Interactive RAG chatbot - ask your own questions.
    """
    print("=" * 60)
    print("hexDAG RAG Chatbot - Interactive Mode")
    print("=" * 60)
    print("Ask questions about hexDAG, LangGraph, or RAG.")
    print("Type 'quit' to exit, 'topics' to see available topics.")
    print()

    while True:
        try:
            question = input("You: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if question.lower() == 'topics':
                print("\nAvailable topics in knowledge base:")
                for doc in KNOWLEDGE_BASE:
                    print(f"  - {doc['title']}")
                print()
                continue

            if not question:
                continue

            print()

            # RAG pipeline
            retrieve_result = await retrieve_context({"question": question})
            generate_result = await generate_answer({
                "question": question,
                "context": retrieve_result["context"]
            })

            print(f"\nBot: {generate_result['answer']}")
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
        await run_rag_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
