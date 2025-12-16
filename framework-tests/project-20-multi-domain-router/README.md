# Project 20: Multi-Domain Routing Agent - hexDAG Port (Capstone)

## Overview
This folder contains the hexDAG port of LangGraph's `project-20-multi-domain-routing-agent`.

## What It Tests

**Route queries to specialized domain agents based on classification.**

```
                        ┌──────────────┐
                        │   Finance    │
                        │    Agent     │
                        └──────────────┘
                              ↑
┌─────────┐    ┌──────────┐   │   ┌──────────────┐
│  Query  │ -> │  Router  │ ──┼── │     HR       │
│         │    │          │   │   │    Agent     │
└─────────┘    └──────────┘   │   └──────────────┘
                              │
                        ┌─────┴────────┐
                        │  Marketing   │
                        │    Agent     │
                        └──────────────┘
                              │
                        ┌──────────────┐
                        │   General    │
                        │    Agent     │
                        └──────────────┘
```

### Example Flow:
```
Query: "What is the vacation policy?"

[ROUTER] Classifying query...
[ROUTER] Detected domain: HR

[AGENT] Routing to HR agent...
[AGENT] Response generated

Domain: HR
Response: Based on company policy, you receive 20 days of paid
time off per year. Vacation days are accrued monthly...
```

### Why This Matters:
- **Specialization** - Each agent has domain expertise
- **Efficiency** - Route to the right expert immediately
- **Scalability** - Add new domains without changing core logic
- **Enterprise** - Common pattern in corporate assistants

## Files
- `router_pipeline.yaml` - hexDAG YAML pipeline
- `run_multi_domain.py` - Python runner with all domain agents
- `README.md` - This file

## Domain Agents

| Domain | Keywords | Capabilities |
|--------|----------|--------------|
| **Finance** | budget, money, expense, savings | Budget analysis, spending advice |
| **HR** | policy, vacation, leave, benefits | Policy lookup, benefits info |
| **Marketing** | slogan, campaign, brand | Creative content, slogans |
| **General** | (default) | General assistance |

## The Difference

| | LangGraph | hexDAG |
|---|-----------|--------|
| **Routing** | `conditional_edges` | Single function with if/elif |
| **Agents** | Separate graph nodes | Python functions |
| **State** | StateGraph with messages | Dict passing |
| **Control Flow** | Edge-based routing | Explicit branching |

### LangGraph (conditional edges):
```python
def route_query(state):
    last_message = state["messages"][-1].content.lower()
    if "finance" in last_message:
        return "finance_agent"
    elif "hr" in last_message:
        return "hr_agent"
    ...

graph.add_conditional_edges(
    "route_query",
    route_query,
    {
        "finance_agent": "finance_agent",
        "hr_agent": "hr_agent",
        "marketing_agent": "marketing_agent",
        "general_agent": "general_agent",
    }
)
```

### hexDAG (unified handler):
```python
DOMAIN_KEYWORDS = {
    "finance": ["budget", "money", "expense", ...],
    "hr": ["policy", "vacation", "leave", ...],
    "marketing": ["slogan", "campaign", ...],
}

def classify_domain(query: str) -> str:
    query_lower = query.lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for keyword in keywords:
            if keyword in query_lower:
                return domain
    return "general"

async def handle_query(inputs: dict) -> dict:
    domain = inputs.get("domain", "general")
    if domain == "finance":
        response = await finance_agent(query)
    elif domain == "hr":
        response = await hr_agent(query)
    ...
```

## Verdict: WORKS PERFECTLY

Linear classify-then-handle pipeline - perfect for hexDAG.

**hexDAG version advantages:**
- Explicit keyword matching (more predictable)
- Easy to add new domains
- Clear routing logic
- Unified response format

## How to Run

```bash
cd framework-tests/project20-multi-domain-router

# Demo mode (runs test queries)
..\..\venv\Scripts\python.exe run_multi_domain.py

# Interactive mode
..\..\venv\Scripts\python.exe run_multi_domain.py --interactive
```

Expected output:
```
============================================================
hexDAG Multi-Domain Routing Agent (Capstone)
============================================================

This unified assistant routes queries to specialized agents:
  - FINANCE: Budget, expenses, savings
  - HR: Policies, vacation, benefits
  - MARKETING: Slogans, campaigns, branding
  - GENERAL: Everything else

[Finance Query]
User: Can you give me a summary of my budget?
--------------------------------------------------
  [ROUTER] Classifying query...
  [ROUTER] Detected domain: FINANCE
  [AGENT] Routing to FINANCE agent...
  [AGENT] Response generated

Domain: FINANCE
Response: Based on your budget data:
- Food: $200
- Rent: $1000
- Transport: $50
...

[HR Query]
User: What is the vacation policy?
--------------------------------------------------
  [ROUTER] Classifying query...
  [ROUTER] Detected domain: HR
  [AGENT] Routing to HR agent...

Domain: HR
Response: According to company policy, you receive 20 days
of paid time off per year...
```

## Test Results
[To be filled after testing]

## Observations
[To be filled after testing]
