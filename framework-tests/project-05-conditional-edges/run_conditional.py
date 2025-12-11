#!/usr/bin/env python
"""
hexDAG Conditional Edges Demo
Ported from LangGraph project-05-multi-agent-collaboration

This demonstrates conditional looping: chatbot responds until
message count exceeds a threshold (5 messages).

LangGraph uses conditional_edges to create cycles.
hexDAG uses external iteration since DAGs are acyclic.

Run with: ..\..\.venv\Scripts\python.exe run_conditional.py
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

# Threshold for stopping the loop (same as LangGraph version)
MESSAGE_THRESHOLD = 5


async def chatbot_node(inputs: dict) -> dict:
    """
    Chatbot node that generates a response.
    Equivalent to the LangGraph chatbot function.
    """
    messages = inputs.get("messages", [])
    user_input = inputs.get("user_input", "")

    model = genai.GenerativeModel("gemini-2.0-flash")

    # Build prompt with conversation history
    prompt_parts = ["You are a helpful assistant.", ""]

    if messages:
        prompt_parts.append("Conversation history:")
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"{role}: {content}")
        prompt_parts.append("")

    prompt_parts.append(f"User: {user_input}")
    prompt_parts.append("")
    prompt_parts.append("Respond helpfully and concisely.")

    full_prompt = "\n".join(prompt_parts)

    # Generate response
    response = model.generate_content(full_prompt)
    response_text = response.text.strip()

    # Update messages
    new_messages = messages.copy()
    new_messages.append({"role": "user", "content": user_input})
    new_messages.append({"role": "assistant", "content": response_text})

    return {
        "response": response_text,
        "messages": new_messages,
        "message_count": len(new_messages)
    }


def check_threshold(message_count: int) -> bool:
    """
    Check if message count exceeds threshold.
    Equivalent to LangGraph's check_threshold function.

    In LangGraph:
        def check_threshold(state):
            return len(state["messages"]) > 5

    Returns True if should STOP, False if should CONTINUE.
    """
    return message_count > MESSAGE_THRESHOLD


async def run_conditional_loop():
    """
    Run the conditional loop workflow.

    LangGraph approach:
        graph.add_conditional_edges("chatbot", check_threshold, {True: "end", False: "chatbot"})
        - This creates a CYCLE that loops until condition is True

    hexDAG approach:
        - DAGs cannot have cycles (Directed ACYCLIC Graph)
        - We implement the loop externally in Python
        - Each iteration runs the chatbot DAG once
        - We check the threshold condition after each run
    """
    print("=" * 60)
    print("hexDAG Conditional Edges Demo")
    print("(Ported from LangGraph)")
    print("=" * 60)
    print(f"Threshold: {MESSAGE_THRESHOLD} messages")
    print("The chatbot will loop until message count > threshold")
    print("=" * 60 + "\n")

    # Create single-node DAG for chatbot
    graph = DirectedGraph()
    graph.add(NodeSpec("chatbot", chatbot_node))

    orchestrator = Orchestrator()

    # Initial state
    messages = []
    iteration = 0

    # Start with initial message (same as LangGraph: "Hello!")
    user_input = "Hello!"

    # Loop until threshold is reached
    # This is the hexDAG equivalent of LangGraph's conditional edges
    while True:
        iteration += 1
        print(f"--- Iteration {iteration} ---")
        print(f"User: {user_input}")

        # Run chatbot DAG
        result = await orchestrator.run(graph, {
            "user_input": user_input,
            "messages": messages
        })

        # Extract results
        chatbot_output = result.get("chatbot", {})
        response = chatbot_output.get("response", "No response")
        messages = chatbot_output.get("messages", [])
        message_count = chatbot_output.get("message_count", 0)

        print(f"Chatbot: {response}")
        print(f"Message count: {message_count}")

        # Check threshold condition (equivalent to conditional_edges)
        if check_threshold(message_count):
            print(f"\n[Threshold reached: {message_count} > {MESSAGE_THRESHOLD}]")
            print("[Exiting loop]")
            break

        # Generate follow-up input for next iteration
        # In the original LangGraph, this would continue automatically
        # Here we simulate continued conversation
        user_input = f"Tell me more (message {iteration + 1})"
        print()

    print("\n" + "=" * 60)
    print("FINAL STATE")
    print("=" * 60)
    print(f"Total iterations: {iteration}")
    print(f"Total messages: {len(messages)}")
    print(f"Last response: {messages[-1]['content'] if messages else 'None'}")


async def main_interactive():
    """
    Interactive version where user provides input each iteration.
    """
    print("=" * 60)
    print("hexDAG Conditional Edges (Interactive Mode)")
    print("=" * 60)
    print(f"Chat will continue until {MESSAGE_THRESHOLD} messages")
    print("Type 'quit' to exit early.\n")

    graph = DirectedGraph()
    graph.add(NodeSpec("chatbot", chatbot_node))
    orchestrator = Orchestrator()

    messages = []

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

        result = await orchestrator.run(graph, {
            "user_input": user_input,
            "messages": messages
        })

        chatbot_output = result.get("chatbot", {})
        response = chatbot_output.get("response", "No response")
        messages = chatbot_output.get("messages", [])
        message_count = len(messages)

        print(f"Chatbot: {response}")
        print(f"[Message count: {message_count}/{MESSAGE_THRESHOLD}]\n")

        if check_threshold(message_count):
            print(f"[Threshold reached! Ending conversation.]")
            break

    print(f"\nFinal message count: {len(messages)}")


if __name__ == "__main__":
    import sys

    print("Choose mode:")
    print("1. Auto-run (simulates LangGraph behavior)")
    print("2. Interactive (you provide input)")
    choice = input("Enter 1 or 2: ").strip()

    try:
        if choice == "1":
            asyncio.run(run_conditional_loop())
        else:
            asyncio.run(main_interactive())
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
