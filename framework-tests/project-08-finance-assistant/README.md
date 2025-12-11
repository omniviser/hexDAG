# Project 8: Finance Assistant - hexDAG Port

## Overview
This folder contains the hexDAG port of LangGraph's `project-08-data-extraction-agent`.

**Note:** The folder name is misleading - this is actually about **financial data queries**, not generic data extraction.

## What It Tests

**Agent with search tool for real-time financial data.**

```
User: "What is the stock price of Apple?"
        |
        v
   [Search Tool]  --> DuckDuckGo: "Apple AAPL stock price"
        |
        v
   [Finance Analyst]  --> LLM analyzes search results
        |
        v
   Response: "Apple (AAPL) is currently trading at $XXX..."
```

### Why This Matters:
- **Real-time data** - LLMs have knowledge cutoffs, search gets current info
- **Tool augmentation** - Combine LLM reasoning with external data sources
- **Domain-specific assistants** - Finance, weather, news, etc.

## Files
- `finance_pipeline.yaml` - hexDAG YAML pipeline
- `run_finance.py` - Python runner with search implementation
- `README.md` - This file

## The Difference

| | LangGraph | hexDAG |
|---|-----------|--------|
| **Search Tool** | TavilySearchResults (paid API) | DuckDuckGo (free) |
| **Tool Integration** | Built-in tool binding | Manual in node function |
| **Flow** | Potential cycle (tool -> chatbot) | Linear DAG |

### LangGraph (built-in tool):
```python
from langchain_community.tools.tavily_search import TavilySearchResults

tool = TavilySearchResults(max_results=2)

# Tool automatically bound to LLM
graph.add_node("tool", lambda state: {"messages": [tool.invoke(...)]})
graph.add_edge("tool", "chatbot")  # Can cycle back
```

### hexDAG (manual tool):
```python
async def search_financial_data(inputs: dict) -> dict:
    # Manual DuckDuckGo search
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))
    return {"search_results": formatted_results}

# Linear flow: search -> analyze
graph.add(NodeSpec("search_tool", search_financial_data))
graph.add(NodeSpec("finance_analyst", finance_analyst, depends_on=["search_tool"]))
```

## Verdict: WORKS WELL

hexDAG handles this pattern well! The linear DAG flow (search -> analyze) is natural.

**Differences:**
- Must implement search tool manually (not built-in)
- No automatic tool binding like LangChain
- DAG ensures predictable flow (no infinite loops)

**Benefits of hexDAG approach:**
- No paid API required (DuckDuckGo is free)
- Clear, predictable execution flow
- Easy to add more tools as nodes

## How to Run

```bash
cd framework-tests/project8-finance-assistant

# Demo mode (runs predefined queries)
..\..\.venv\Scripts\python.exe run_finance.py

# Interactive mode (ask your own questions)
..\..\.venv\Scripts\python.exe run_finance.py --interactive
```

Expected output:
```
============================================================
hexDAG Finance Assistant Demo
============================================================

[Query 1] What is the current stock price of Apple AAPL?
--------------------------------------------------
  [SEARCH] Searching for: What is the current stock price of Apple AAPL?
  [SEARCH] Found 5 results
  [ANALYST] Analyzing results...

Response:
Based on the search results, Apple (AAPL) is currently trading at...

============================================================
```

## Test Results
[To be filled after testing]

## Observations
[To be filled after testing]
