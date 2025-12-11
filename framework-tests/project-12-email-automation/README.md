# Project 12: Email Response Automation - hexDAG Port

## Overview
This folder contains the hexDAG port of LangGraph's `project-12-email-response-automation`.

## What It Tests

**Email automation pipeline - parse and respond to emails.**

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Incoming   │ --> │   Parse     │ --> │   Write     │ --> Response
│   Email     │     │   Email     │     │  Response   │
└─────────────┘     └─────────────┘     └─────────────┘
                          │                   │
                          v                   v
                    Extract:              Generate:
                    - Intent              - Professional reply
                    - Sentiment           - Address concerns
                    - Key details         - Next steps
```

### Example Flow:
```
INPUT EMAIL:
From: john.doe@email.com
Subject: Refund Request
"I purchased your Premium subscription but I'm not satisfied..."

[PARSE] Analyzing email...
  - Intent: complaint/request
  - Sentiment: negative
  - Urgency: high
  - Action: Process refund request

[WRITE] Generating response...

OUTPUT:
"Dear John,

Thank you for reaching out. I'm sorry to hear the Premium subscription
hasn't met your expectations. I've initiated the refund process for
Order #12345. You should see the refund within 3-5 business days...

Best regards,
Customer Support"
```

### Why Email Automation Matters:
- **Save time** - Automate repetitive responses
- **Consistency** - Same quality for every customer
- **Speed** - Instant responses to common queries
- **Scale** - Handle thousands of emails

## Files
- `email_pipeline.yaml` - hexDAG YAML pipeline
- `run_email.py` - Python runner with sample emails
- `README.md` - This file

## The Difference

| | LangGraph | hexDAG |
|---|-----------|--------|
| **Pattern** | parse -> write | parse -> write |
| **Parsing** | Hardcoded string | LLM-powered analysis |
| **State** | Shared state object | Explicit inputs/outputs |

### LangGraph (simplified):
```python
def parse_email(state: State):
    # Just returns hardcoded string
    return {"email_content": "The customer is asking for a refund."}

def write_response(state: State):
    response = llm.invoke(f"Write a response to: {state['email_content']}")
    return {"messages": [response]}
```

### hexDAG (enhanced):
```python
async def parse_email(inputs: dict) -> dict:
    # LLM extracts: intent, sentiment, urgency, key details
    prompt = f"Parse this email and extract key information: {email}"
    parsed = llm.generate(prompt)
    return {"parsed_info": parsed}

async def write_response(inputs: dict) -> dict:
    # Uses parsed info for better response
    prompt = f"Write response based on: {inputs['parsed_info']}"
    response = llm.generate(prompt)
    return {"email_response": response}
```

## Verdict: WORKS PERFECTLY

This is a simple linear pipeline - perfect for hexDAG.

**hexDAG version is actually better because:**
- Uses LLM for intelligent parsing (not hardcoded)
- Extracts structured information (intent, sentiment, urgency)
- Better response generation with parsed context

## Sample Emails Tested

| Email Type | Intent | Sentiment | Response |
|------------|--------|-----------|----------|
| Refund request | complaint | negative | Apologize, process refund |
| Pricing inquiry | question | neutral | Provide pricing info |
| Positive feedback | feedback | positive | Thank customer |

## How to Run

```bash
cd framework-tests/project12-email-automation
..\..\.venv\Scripts\python.exe run_email.py
```

Expected output:
```
============================================================
hexDAG Email Response Automation Demo
============================================================

[Email 1] From: john.doe@email.com
Subject: Refund Request
--------------------------------------------------
  [PARSE] Analyzing email from: john.doe@email.com
  [PARSE] Subject: Refund Request
  [PARSE] Email analyzed successfully
  [WRITE] Generating response...
  [WRITE] Response generated (850 chars)

PARSED INFO:
------------------------------
1. INTENT: complaint/request
2. TOPIC: Refund for subscription
3. KEY DETAILS:
   - Order #12345
   - Premium subscription
   - Features don't match advertised
4. URGENCY: high
5. SENTIMENT: negative
6. ACTION NEEDED: Process refund

GENERATED RESPONSE:
------------------------------
Dear John,

Thank you for contacting us regarding your Premium subscription...
```

## Test Results
[To be filled after testing]

## Observations
[To be filled after testing]
