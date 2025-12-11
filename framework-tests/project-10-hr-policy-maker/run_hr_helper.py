#!/usr/bin/env python
"""
hexDAG HR Policy Helper Demo
Ported from LangGraph project-10-decision-making-agent

Pattern: RAG (Retrieval Augmented Generation)
- Store policy documents in a knowledge base
- Retrieve relevant documents based on question
- LLM generates answer using retrieved context

Run with: ..\..\.venv\Scripts\python.exe run_hr_helper.py
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


# Sample HR Policy Documents (in real app, this would be a database/vector store)
HR_POLICIES = [
    {
        "id": "vacation",
        "title": "Vacation Policy",
        "content": """
        VACATION POLICY
        - Full-time employees receive 20 days of paid time off (PTO) per year
        - PTO accrues at 1.67 days per month
        - Unused PTO can be carried over up to 5 days to the next year
        - PTO requests must be submitted at least 2 weeks in advance
        - Manager approval is required for all PTO requests
        """
    },
    {
        "id": "sick_leave",
        "title": "Sick Leave Policy",
        "content": """
        SICK LEAVE POLICY
        - Employees receive 10 days of paid sick leave per year
        - Sick leave can be used for personal illness or family care
        - Doctor's note required for absences of 3+ consecutive days
        - Unused sick leave does not carry over
        """
    },
    {
        "id": "remote_work",
        "title": "Remote Work Policy",
        "content": """
        REMOTE WORK POLICY
        - Employees may work remotely up to 3 days per week
        - Core hours are 10am-3pm in your local timezone
        - Must be available for video calls during core hours
        - Home office setup stipend: $500 one-time
        - Internet reimbursement: $50/month
        """
    },
    {
        "id": "benefits",
        "title": "Benefits Overview",
        "content": """
        EMPLOYEE BENEFITS
        - Health insurance: Company covers 80% of premium
        - Dental and vision: Company covers 100%
        - 401(k): 4% company match
        - Life insurance: 2x annual salary
        - Professional development: $2,000/year budget
        """
    },
    {
        "id": "parental_leave",
        "title": "Parental Leave Policy",
        "content": """
        PARENTAL LEAVE POLICY
        - Primary caregiver: 16 weeks paid leave
        - Secondary caregiver: 6 weeks paid leave
        - Can be taken within 12 months of birth/adoption
        - Flexible return options available
        """
    }
]


def simple_keyword_search(query: str, documents: list, top_k: int = 3) -> list:
    """
    Simple keyword-based document retrieval.

    In production, you'd use:
    - FAISS (like LangGraph example)
    - Pinecone, Weaviate, Chroma
    - Elasticsearch

    This simple version just matches keywords for demo purposes.
    """
    query_words = set(query.lower().split())

    scored_docs = []
    for doc in documents:
        # Count matching words in title and content
        doc_text = (doc["title"] + " " + doc["content"]).lower()
        doc_words = set(doc_text.split())

        # Simple relevance score: count of matching words
        score = len(query_words.intersection(doc_words))

        # Boost score for exact phrase matches
        if any(word in doc_text for word in query_words):
            score += 2

        scored_docs.append((score, doc))

    # Sort by score and return top_k
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_docs[:top_k] if score > 0]


async def retrieve_policies(inputs: dict) -> dict:
    """
    Retrieve relevant policy documents based on question.

    LangGraph version uses FAISS vector store:
        vectorstore = FAISS.from_texts(texts, embeddings)
        retriever = vectorstore.as_retriever()
        docs = retriever.invoke(question)

    hexDAG version uses simple keyword matching (for demo)
    or can use the same FAISS setup.
    """
    question = inputs.get("question", "")

    print(f"  [RETRIEVE] Searching policies for: {question}")

    # Find relevant documents
    relevant_docs = simple_keyword_search(question, HR_POLICIES, top_k=2)

    if relevant_docs:
        # Format retrieved documents
        context_parts = []
        for doc in relevant_docs:
            context_parts.append(f"=== {doc['title']} ===\n{doc['content']}")

        retrieved_context = "\n\n".join(context_parts)
        print(f"  [RETRIEVE] Found {len(relevant_docs)} relevant policies")
    else:
        retrieved_context = "No relevant policy documents found."
        print(f"  [RETRIEVE] No matching policies found")

    return {
        "retrieved_context": retrieved_context,
        "num_docs_found": len(relevant_docs)
    }


async def hr_assistant(inputs: dict) -> dict:
    """
    Generate answer using retrieved policy context.

    This is the "Generation" part of RAG - the LLM uses
    the retrieved documents to answer the question.
    """
    question = inputs.get("question", "")
    context = inputs.get("retrieved_context", "No context available")

    print(f"  [ASSISTANT] Generating answer...")

    prompt = f"""You are an HR Policy Assistant. Answer the employee's question
based ONLY on the provided policy documents.

Employee Question: {question}

Relevant Policy Documents:
{context}

Instructions:
- Only use information from the policy documents
- If the answer isn't in the documents, say "I don't have information about that in our policy documents"
- Be helpful and professional
- Keep your answer concise"""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    return {
        "answer": response.text.strip(),
        "question": question
    }


async def run_hr_demo():
    """
    Demonstrate HR Policy Helper with RAG pattern.
    """
    print("=" * 60)
    print("hexDAG HR Policy Helper Demo")
    print("=" * 60)
    print()

    # Create graph
    graph = DirectedGraph()
    graph.add(NodeSpec("retrieve", retrieve_policies))
    graph.add(NodeSpec("assistant", hr_assistant, depends_on=["retrieve"]))

    orchestrator = Orchestrator()

    # Test questions
    questions = [
        "How many vacation days do I get?",
        "What is the remote work policy?",
        "How much does the company match for 401k?",
        "What is the parental leave policy?",
        "Can I bring my dog to the office?",  # Not in policies
    ]

    for i, question in enumerate(questions, 1):
        print(f"[Question {i}] {question}")
        print("-" * 50)

        # Run retrieval
        retrieve_result = await retrieve_policies({"question": question})

        # Run assistant with retrieved context
        assistant_input = {
            "question": question,
            "retrieved_context": retrieve_result["retrieved_context"]
        }
        assistant_result = await hr_assistant(assistant_input)

        print(f"\nAnswer: {assistant_result['answer']}")
        print()
        print("=" * 60)
        print()


async def interactive_mode():
    """
    Interactive HR assistant - ask your own questions.
    """
    print("=" * 60)
    print("hexDAG HR Policy Helper - Interactive Mode")
    print("=" * 60)
    print("Ask questions about company policies.")
    print("Type 'quit' to exit, 'policies' to see available topics.")
    print()

    while True:
        try:
            question = input("Employee: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if question.lower() == 'policies':
                print("\nAvailable policy topics:")
                for policy in HR_POLICIES:
                    print(f"  - {policy['title']}")
                print()
                continue

            if not question:
                continue

            print()

            # Retrieve and answer
            retrieve_result = await retrieve_policies({"question": question})
            assistant_input = {
                "question": question,
                "retrieved_context": retrieve_result["retrieved_context"]
            }
            assistant_result = await hr_assistant(assistant_input)

            print(f"\nHR Assistant: {assistant_result['answer']}")
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
        await run_hr_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
