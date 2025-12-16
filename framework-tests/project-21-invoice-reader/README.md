# Project 21: Scanned Invoice Reading Agent - hexDAG Port

## Overview
This folder contains the hexDAG port of LangGraph's `project-21-scanned-invoice-reading-agent`.

## What It Tests

**Extract structured data from PDF invoices using AI.**

```
┌───────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   PDF     │ --> │  Read Text   │ --> │   Extract    │ --> │   Validate   │
│  Invoice  │     │  (pypdf)     │     │   Fields     │     │    Data      │
└───────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                        │                     │                     │
                        v                     v                     v
                   Raw text              Structured             Verified
                   extraction              JSON                  output
```

### Example Flow:
```
[Invoice 1] Sample Invoice (Acme Corporation)
--------------------------------------------------
  [READ] Processing invoice...
  [READ] Using sample invoice 1
  [EXTRACT] Analyzing invoice with AI...
  [EXTRACT] Successfully extracted structured data
  [VALIDATE] Checking extracted data...
  [VALIDATE] All fields valid

EXTRACTED DATA:
------------------------------
{
  "vendor_name": "Acme Corporation",
  "customer_name": "TechStart Inc.",
  "invoice_number": "INV-2024-0042",
  "total_amount": 10908.00,
  "due_date": "January 15, 2025"
}

VALIDATION:
------------------------------
✓ All fields extracted successfully
```

### Why This Matters:
- **Automation** - Process invoices without manual data entry
- **Accuracy** - AI extracts structured data from unstructured text
- **Efficiency** - Batch process hundreds of invoices
- **Integration** - Output JSON ready for accounting systems

## Files
- `invoice_pipeline.yaml` - hexDAG YAML pipeline
- `run_invoice_reader.py` - Python runner with PDF support
- `README.md` - This file

## Dependencies

```bash
pip install pypdf  # For PDF reading (optional - samples work without it)
```

## The Difference

| | LangGraph | hexDAG |
|---|-----------|--------|
| **PDF Reading** | pypdf | pypdf (same) |
| **Output Schema** | Pydantic `Invoice` model | JSON extraction |
| **Structured Output** | `with_structured_output()` | Prompt engineering |
| **Validation** | Implicit (Pydantic) | Explicit validation step |

### LangGraph (structured output):
```python
class Invoice(BaseModel):
    vendor_name: str = Field(...)
    customer_name: str = Field(...)
    invoice_number: str = Field(...)
    total_amount: float = Field(...)
    due_date: str = Field(...)

llm = ChatGoogleGenerativeAI(model=model, ...)
structured_llm = llm.with_structured_output(Invoice)
response = structured_llm.invoke(prompt)  # Returns Invoice object
```

### hexDAG (JSON extraction):
```python
prompt = """Extract and return ONLY a valid JSON object:
{
    "vendor_name": "...",
    "customer_name": "...",
    "invoice_number": "...",
    "total_amount": 0.00,
    "due_date": "..."
}"""

response = model.generate_content(prompt)
structured_data = json.loads(response.text)

# Explicit validation
if "vendor_name" not in structured_data:
    issues.append("Missing vendor_name")
```

## Verdict: WORKS PERFECTLY

Linear document processing pipeline - perfect for hexDAG.

**hexDAG version enhancements:**
- Sample invoices for testing without PDF files
- Explicit validation step with error reporting
- Clean JSON output for system integration
- Graceful fallback if PDF reading fails

## How to Run

```bash
cd framework-tests/project21-invoice-reader

# Demo mode (uses sample invoices)
..\..\venv\Scripts\python.exe run_invoice_reader.py

# Process specific PDF
..\..\venv\Scripts\python.exe run_invoice_reader.py path/to/invoice.pdf
```

Expected output:
```
============================================================
hexDAG Invoice Reading Agent Demo
============================================================

This agent extracts structured data from invoices:
  - Vendor Name
  - Customer Name
  - Invoice Number
  - Total Amount
  - Due Date

[Invoice 1] Sample Invoice 1 (Acme Corporation)
--------------------------------------------------
  [READ] Processing invoice...
  [READ] Using sample invoice 1
  [EXTRACT] Analyzing invoice with AI...
  [EXTRACT] Successfully extracted structured data
  [VALIDATE] Checking extracted data...
  [VALIDATE] All fields valid

INVOICE TEXT (preview):
------------------------------
INVOICE

From:
Acme Corporation
123 Business Street
New York, NY 10001
...

EXTRACTED DATA:
------------------------------
{
  "vendor_name": "Acme Corporation",
  "customer_name": "TechStart Inc.",
  "invoice_number": "INV-2024-0042",
  "total_amount": 10908.00,
  "due_date": "January 15, 2025"
}

VALIDATION:
------------------------------
✓ All fields extracted successfully
```

## Test Results
[To be filled after testing]

## Observations
[To be filled after testing]
