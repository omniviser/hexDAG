#!/usr/bin/env python
"""
hexDAG Chatbot Runner
Ported from LangGraph project-01-basic-chatbot
Uses Google Gemini API (same as original)

Run with: ..\..\venv\Scripts\python.exe run_chatbot.py
Or from project root: .venv\Scripts\python.exe framework-tests\project1-chatbot\run_chatbot.py
"""
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load environment variables from the langgraph-tutorials .env file
env_path = project_root / "reference_examples" / "langgraph-tutorials" / ".env"
load_dotenv(env_path)

import google.generativeai as genai
from hexdag.core.domain.dag import DirectedGraph, NodeSpec
from hexdag.core.orchestration.orchestrator import Orchestrator


# Configure Google Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY not found in .env file")
    print(f"Looked in: {env_path}")
    sys.exit(1)

genai.configure(api_key=GEMINI_API_KEY)


async def chatbot(inputs: dict) -> dict:
    """
    Chatbot node that uses Google Gemini to generate responses.
    Equivalent to the LangGraph chatbot function.
    """
    # Get conversation history and current user input
    messages = inputs.get("messages", [])
    user_input = inputs.get("user_input", "")

    # Build conversation history for Gemini
    # Gemini uses a different format than LangChain
    model = genai.GenerativeModel("gemini-2.0-flash")

    # Build the prompt with conversation history
    prompt_parts = []

    if messages:
        prompt_parts.append("Previous conversation:")
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"{role}: {content}")
        prompt_parts.append("")

    prompt_parts.append(f"User: {user_input}")
    prompt_parts.append("")
    prompt_parts.append("Respond naturally and helpfully.")

    full_prompt = "\n".join(prompt_parts)

    # Generate response
    response = model.generate_content(full_prompt)

    # Extract text from response
    response_text = response.text

    # Update messages history
    messages.append({"role": "user", "content": user_input})
    messages.append({"role": "assistant", "content": response_text})

    return {
        "response": response_text,
        "messages": messages
    }


async def main():
    """Main interaction loop - mirrors the LangGraph version."""
    print("=" * 60)
    print("hexDAG Chatbot (Ported from LangGraph)")
    print("Using Google Gemini API")
    print("=" * 60)
    print("Type 'quit' to exit.\n")

    # Create single-node graph (same structure as LangGraph version)
    graph = DirectedGraph()
    graph.add(NodeSpec("chatbot", chatbot))

    # Create orchestrator
    orchestrator = Orchestrator()

    # Maintain conversation history (equivalent to LangGraph state)
    conversation_messages = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break

        # Run the DAG (equivalent to app.invoke in LangGraph)
        result = await orchestrator.run(graph, {
            "user_input": user_input,
            "messages": conversation_messages
        })

        # Extract response (equivalent to response["messages"][-1].content)
        chatbot_output = result.get("chatbot", {})
        response = chatbot_output.get("response", "No response")
        conversation_messages = chatbot_output.get("messages", [])

        print(f"Chatbot: {response}\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
