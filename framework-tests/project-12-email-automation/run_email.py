#!/usr/bin/env python
"""
hexDAG Email Response Automation Demo
Ported from LangGraph project-12-email-response-automation

Pattern: Email Automation Pipeline
- Parse incoming email to extract key information
- Generate appropriate professional response

Run with: ..\..\.venv\Scripts\python.exe run_email.py
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


# Sample emails for testing
SAMPLE_EMAILS = [
    {
        "id": 1,
        "subject": "Refund Request",
        "from": "john.doe@email.com",
        "body": """
Hi,

I purchased your Premium subscription last week (Order #12345) but I'm not
satisfied with the service. The features don't match what was advertised
on your website.

I would like to request a full refund. Please process this as soon as possible.

Thanks,
John
"""
    },
    {
        "id": 2,
        "subject": "Question about pricing",
        "from": "sarah.smith@company.com",
        "body": """
Hello,

I'm interested in your Enterprise plan for my team of 50 people.
Could you provide information about:
- Volume discounts
- Annual vs monthly pricing
- What features are included

We're looking to make a decision by end of month.

Best regards,
Sarah Smith
Head of Operations
"""
    },
    {
        "id": 3,
        "subject": "Great product!",
        "from": "happy.customer@gmail.com",
        "body": """
Just wanted to say your product is amazing! It's saved our team
so much time. The customer support has been fantastic too.

Keep up the great work!

- Mike
"""
    },
]


async def parse_email(inputs: dict) -> dict:
    """
    Parse email to extract key information.

    LangGraph version (simplified):
        def parse_email(state):
            return {"email_content": "The customer is asking for a refund."}

    hexDAG version: Uses LLM for intelligent parsing.
    """
    email_content = inputs.get("email_content", "")
    email_subject = inputs.get("email_subject", "")
    email_from = inputs.get("email_from", "")

    print(f"  [PARSE] Analyzing email from: {email_from}")
    print(f"  [PARSE] Subject: {email_subject}")

    prompt = f"""Parse this email and extract key information:

FROM: {email_from}
SUBJECT: {email_subject}

EMAIL BODY:
{email_content}

Extract and format as follows:
1. INTENT: (question/complaint/request/feedback/inquiry)
2. TOPIC: (main subject in 5 words or less)
3. KEY DETAILS: (bullet points of important info)
4. URGENCY: (low/medium/high)
5. SENTIMENT: (positive/neutral/negative)
6. ACTION NEEDED: (what response is required)"""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    parsed_info = response.text.strip()
    print(f"  [PARSE] Email analyzed successfully")

    return {
        "parsed_info": parsed_info,
        "email_from": email_from,
        "email_subject": email_subject
    }


async def write_response(inputs: dict) -> dict:
    """
    Generate appropriate email response.

    LangGraph version:
        def write_response(state):
            response = llm.invoke(f"Write a response to this email: {state['email_content']}")
            return {"messages": [response]}
    """
    email_content = inputs.get("email_content", "")
    parsed_info = inputs.get("parsed_info", "")
    email_from = inputs.get("email_from", "")
    email_subject = inputs.get("email_subject", "")

    print(f"  [WRITE] Generating response...")

    prompt = f"""Write a professional email response.

ORIGINAL EMAIL:
From: {email_from}
Subject: {email_subject}
Body: {email_content}

PARSED ANALYSIS:
{parsed_info}

Write a response that:
- Is professional and friendly
- Addresses all concerns/questions raised
- Provides clear next steps if needed
- Is concise but complete
- Includes appropriate greeting and sign-off

Format as a complete email response."""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    email_response = response.text.strip()
    print(f"  [WRITE] Response generated ({len(email_response)} chars)")

    return {
        "email_response": email_response,
        "original_subject": email_subject
    }


async def run_email_demo():
    """
    Demonstrate email parsing and response generation.
    """
    print("=" * 60)
    print("hexDAG Email Response Automation Demo")
    print("=" * 60)
    print()

    # Create graph
    graph = DirectedGraph()
    graph.add(NodeSpec("parse", parse_email))
    graph.add(NodeSpec("write", write_response, depends_on=["parse"]))

    orchestrator = Orchestrator()

    for email in SAMPLE_EMAILS:
        print(f"[Email {email['id']}] From: {email['from']}")
        print(f"Subject: {email['subject']}")
        print("-" * 50)

        # Parse the email
        parse_input = {
            "email_content": email["body"],
            "email_subject": email["subject"],
            "email_from": email["from"]
        }
        parse_result = await parse_email(parse_input)

        # Generate response
        write_input = {
            "email_content": email["body"],
            "parsed_info": parse_result["parsed_info"],
            "email_from": email["from"],
            "email_subject": email["subject"]
        }
        write_result = await write_response(write_input)

        # Display results
        print()
        print("PARSED INFO:")
        print("-" * 30)
        print(parse_result["parsed_info"])
        print()
        print("GENERATED RESPONSE:")
        print("-" * 30)
        print(write_result["email_response"])
        print()
        print("=" * 60)
        print()


async def process_single_email(email_body: str, subject: str = "No Subject", sender: str = "unknown@email.com"):
    """
    Process a single email.
    """
    print(f"Processing email from: {sender}")
    print("-" * 50)

    parse_result = await parse_email({
        "email_content": email_body,
        "email_subject": subject,
        "email_from": sender
    })

    write_result = await write_response({
        "email_content": email_body,
        "parsed_info": parse_result["parsed_info"],
        "email_from": sender,
        "email_subject": subject
    })

    print()
    print("RESPONSE:")
    print("=" * 50)
    print(write_result["email_response"])

    return write_result


async def main():
    """Main entry point."""
    await run_email_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
