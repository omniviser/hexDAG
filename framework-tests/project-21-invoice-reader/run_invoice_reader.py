#!/usr/bin/env python
"""
hexDAG Scanned Invoice Reading Agent Demo
Ported from LangGraph project-21-scanned-invoice-reading-agent

Pattern: Document Processing + Structured Extraction
- Read PDF invoice text
- Extract structured data using LLM
- Output clean JSON with invoice fields

Run with: ..\..\venv\Scripts\python.exe run_invoice_reader.py
"""
import asyncio
import json
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

# Configure Google Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY not found in .env file")
    sys.exit(1)

genai.configure(api_key=GEMINI_API_KEY)

# Try to import pypdf for PDF reading
try:
    from pypdf import PdfReader
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("WARNING: pypdf not installed. Install with: pip install pypdf")
    print("         Will use sample invoice text for demo.\n")


# Sample invoice data for testing (when PDF not available)
SAMPLE_INVOICE_TEXT = """
INVOICE

From:
Acme Corporation
123 Business Street
New York, NY 10001

Bill To:
TechStart Inc.
456 Innovation Ave
San Francisco, CA 94102

Invoice Number: INV-2024-0042
Invoice Date: December 15, 2024
Due Date: January 15, 2025

Description                          Quantity    Unit Price    Amount
------------------------------------------------------------------------
Software Development Services            40        $150.00    $6,000.00
Cloud Infrastructure Setup                1      $2,500.00    $2,500.00
Technical Consultation                    8        $200.00    $1,600.00

------------------------------------------------------------------------
                                              Subtotal:      $10,100.00
                                              Tax (8%):         $808.00
                                              --------------------------
                                              TOTAL DUE:     $10,908.00

Payment Terms: Net 30
Please make payment to: Acme Corp, Account #1234567890

Thank you for your business!
"""

SAMPLE_INVOICE_2 = """
INVOICE #7891

Vendor: Global Supplies Ltd.
Address: 789 Commerce Blvd, Chicago, IL 60601

Customer: DataFlow Systems
Address: 321 Tech Park, Austin, TX 78701

Date: December 10, 2024
Payment Due: December 25, 2024

Items:
- Office Supplies Bundle: $450.00
- Computer Equipment: $3,200.00
- Software Licenses (Annual): $1,800.00

Total Amount: $5,450.00

Payment Method: Bank Transfer
Account: 9876543210
"""


def read_pdf_invoice(pdf_path: str) -> str:
    """
    Read text from a PDF invoice file.

    LangGraph version:
        reader = PdfReader(state["invoice_path"])
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    """
    if not PDF_SUPPORT:
        raise ImportError("pypdf not installed")

    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"

    return text.strip()


async def read_invoice(inputs: dict) -> dict:
    """
    Read invoice text from PDF or use sample data.
    """
    pdf_path = inputs.get("pdf_path")
    use_sample = inputs.get("use_sample", False)
    sample_index = inputs.get("sample_index", 0)

    print(f"  [READ] Processing invoice...")

    if use_sample or not pdf_path:
        # Use sample invoice text
        samples = [SAMPLE_INVOICE_TEXT, SAMPLE_INVOICE_2]
        invoice_text = samples[sample_index % len(samples)]
        print(f"  [READ] Using sample invoice {sample_index + 1}")
    else:
        # Read from PDF file
        try:
            invoice_text = read_pdf_invoice(pdf_path)
            print(f"  [READ] Successfully read PDF: {pdf_path}")
        except Exception as e:
            print(f"  [READ] Error reading PDF: {e}")
            print(f"  [READ] Falling back to sample invoice")
            invoice_text = SAMPLE_INVOICE_TEXT

    return {
        "invoice_text": invoice_text,
        "source": pdf_path if pdf_path and not use_sample else "sample"
    }


async def extract_invoice_data(inputs: dict) -> dict:
    """
    Extract structured data from invoice text using LLM.

    LangGraph version:
        llm = ChatGoogleGenerativeAI(model=model, ...)
        structured_llm = llm.with_structured_output(Invoice)
        response = structured_llm.invoke(prompt)

    hexDAG version: Direct Gemini API with JSON extraction.
    """
    invoice_text = inputs.get("invoice_text", "")

    print(f"  [EXTRACT] Analyzing invoice with AI...")

    prompt = f"""You are an expert accounting assistant.
Analyze the following invoice text and extract key details.

Invoice Text:
---
{invoice_text}
---

Extract and return ONLY a valid JSON object with these exact fields:
{{
    "vendor_name": "company issuing the invoice",
    "customer_name": "customer on the invoice",
    "invoice_number": "unique invoice identifier",
    "total_amount": 0.00,
    "due_date": "date as written on invoice"
}}

IMPORTANT: Return ONLY the JSON object, no markdown, no explanation, just pure JSON."""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    # Parse JSON from response
    response_text = response.text.strip()

    # Clean up response if needed (remove markdown code blocks)
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        response_text = "\n".join(lines[1:-1])

    try:
        structured_data = json.loads(response_text)
        print(f"  [EXTRACT] Successfully extracted structured data")
    except json.JSONDecodeError as e:
        print(f"  [EXTRACT] JSON parsing error: {e}")
        print(f"  [EXTRACT] Raw response: {response_text[:200]}...")
        structured_data = {
            "vendor_name": "Parse Error",
            "customer_name": "Parse Error",
            "invoice_number": "Parse Error",
            "total_amount": 0.0,
            "due_date": "Parse Error",
            "raw_response": response_text
        }

    return {
        "structured_data": structured_data,
        "invoice_text": invoice_text
    }


async def validate_extraction(inputs: dict) -> dict:
    """
    Validate the extracted data and flag any issues.
    """
    data = inputs.get("structured_data", {})

    print(f"  [VALIDATE] Checking extracted data...")

    issues = []

    # Check required fields
    required_fields = ["vendor_name", "customer_name", "invoice_number", "total_amount", "due_date"]
    for field in required_fields:
        if field not in data:
            issues.append(f"Missing field: {field}")
        elif data[field] in ["Parse Error", "", None]:
            issues.append(f"Invalid value for: {field}")

    # Validate total_amount is numeric
    if "total_amount" in data:
        try:
            amount = float(data["total_amount"])
            if amount < 0:
                issues.append("Negative total_amount")
        except (ValueError, TypeError):
            issues.append("total_amount is not a valid number")

    validation_result = {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "structured_data": data
    }

    if issues:
        print(f"  [VALIDATE] Found {len(issues)} issue(s)")
    else:
        print(f"  [VALIDATE] All fields valid")

    return validation_result


async def run_invoice_demo():
    """
    Demonstrate invoice reading and extraction.
    """
    print("=" * 60)
    print("hexDAG Invoice Reading Agent Demo")
    print("=" * 60)
    print()
    print("This agent extracts structured data from invoices:")
    print("  - Vendor Name")
    print("  - Customer Name")
    print("  - Invoice Number")
    print("  - Total Amount")
    print("  - Due Date")
    print()

    # Check for actual PDF file
    script_dir = Path(__file__).parent
    langgraph_dir = project_root / "reference_examples" / "langgraph-tutorials" / "project-21-scanned-invoice-reading-agent"
    pdf_path = langgraph_dir / "invoice01.pdf"

    # Test cases
    test_cases = []

    # Add PDF if available
    if pdf_path.exists() and PDF_SUPPORT:
        test_cases.append({
            "name": "PDF Invoice (from LangGraph example)",
            "pdf_path": str(pdf_path),
            "use_sample": False
        })

    # Add sample invoices
    test_cases.extend([
        {
            "name": "Sample Invoice 1 (Acme Corporation)",
            "use_sample": True,
            "sample_index": 0
        },
        {
            "name": "Sample Invoice 2 (Global Supplies)",
            "use_sample": True,
            "sample_index": 1
        }
    ])

    for i, test in enumerate(test_cases, 1):
        print(f"[Invoice {i}] {test['name']}")
        print("-" * 50)

        # Step 1: Read invoice
        read_result = await read_invoice(test)

        # Step 2: Extract structured data
        extract_result = await extract_invoice_data({
            "invoice_text": read_result["invoice_text"]
        })

        # Step 3: Validate
        validation = await validate_extraction({
            "structured_data": extract_result["structured_data"]
        })

        # Display results
        print()
        print("INVOICE TEXT (preview):")
        print("-" * 30)
        preview = read_result["invoice_text"][:300]
        if len(read_result["invoice_text"]) > 300:
            preview += "\n... (truncated)"
        print(preview)
        print()

        print("EXTRACTED DATA:")
        print("-" * 30)
        print(json.dumps(extract_result["structured_data"], indent=2))
        print()

        print("VALIDATION:")
        print("-" * 30)
        if validation["is_valid"]:
            print("✓ All fields extracted successfully")
        else:
            print("⚠ Issues found:")
            for issue in validation["issues"]:
                print(f"  - {issue}")
        print()
        print("=" * 60)
        print()


async def process_single_invoice(pdf_path: str = None):
    """
    Process a single invoice (PDF or sample).
    """
    print("=" * 60)
    print("hexDAG Invoice Processing")
    print("=" * 60)
    print()

    # Read
    if pdf_path and Path(pdf_path).exists() and PDF_SUPPORT:
        read_result = await read_invoice({"pdf_path": pdf_path})
    else:
        if pdf_path:
            print(f"Note: Could not read '{pdf_path}', using sample invoice")
        read_result = await read_invoice({"use_sample": True})

    # Extract
    extract_result = await extract_invoice_data({
        "invoice_text": read_result["invoice_text"]
    })

    # Validate
    validation = await validate_extraction({
        "structured_data": extract_result["structured_data"]
    })

    # Output
    print()
    print("FINAL RESULT:")
    print("=" * 60)
    print(json.dumps({
        "source": read_result.get("source", "unknown"),
        "valid": validation["is_valid"],
        "data": extract_result["structured_data"]
    }, indent=2))

    return extract_result["structured_data"]


async def main():
    """Main entry point."""
    import sys

    if len(sys.argv) > 1:
        # Process specific PDF file
        pdf_path = sys.argv[1]
        await process_single_invoice(pdf_path)
    else:
        # Run demo with samples
        await run_invoice_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
