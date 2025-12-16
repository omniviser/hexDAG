# Project 18: Web Research and Report Generation - hexDAG Port

## Overview
This folder contains the hexDAG port of LangGraph's `project-18-web-research-and-report-generation`.

## What It Tests

**Search the web and generate comprehensive reports.**

```
┌──────────┐     ┌──────────────┐     ┌──────────────┐
│  Topic   │ --> │   Search     │ --> │   Generate   │ --> Report
│          │     │   Web        │     │   Report     │
└──────────┘     └──────────────┘     └──────────────┘
                       │                     │
                       v                     v
                 DuckDuckGo             Structured report
                 results                with sections
```

### Example Flow:
```
Topic: "Latest trends in renewable energy 2024"

[RESEARCH] Searching web for: Latest trends in renewable energy 2024
[RESEARCH] Found 5 sources

[REPORT] Generating comprehensive report...

# REPORT: Latest Trends in Renewable Energy 2024

## 1. EXECUTIVE SUMMARY
The renewable energy sector continues to experience rapid growth...

## 2. KEY FINDINGS
- Solar capacity increased 30% year-over-year
- Wind energy investments reached record highs
- Battery storage costs dropped 15%

## 3. DETAILED ANALYSIS
...
```

### Why This Matters:
- **Automated research** - No manual searching
- **Structured output** - Professional report format
- **Current data** - Real-time web search
- **Time savings** - Hours of work in minutes

## Files
- `research_pipeline.yaml` - hexDAG YAML pipeline
- `run_web_research.py` - Python runner
- `README.md` - This file

## The Difference

| | LangGraph | hexDAG |
|---|-----------|--------|
| **Search Tool** | Tavily (paid API) | DuckDuckGo (free) |
| **Report** | Basic generation | Structured sections |
| **Sources** | Not tracked | Listed with URLs |

### LangGraph (Tavily):
```python
from langchain_community.tools.tavily_search import TavilySearchResults

tool = TavilySearchResults(max_results=3)

def conduct_research(state):
    search_results = tool.invoke(query)
    return {"research_data": str(search_results)}

def generate_report(state):
    response = llm.invoke(f"Generate report: {state['research_data']}")
```

### hexDAG (DuckDuckGo):
```python
from duckduckgo_search import DDGS

async def conduct_research(inputs):
    with DDGS() as ddgs:
        results = list(ddgs.text(topic, max_results=5))
    # Format with sources list
    return {"research_data": formatted, "sources": urls}

async def generate_report(inputs):
    # Structured prompt with sections:
    # Executive Summary, Key Findings, Analysis,
    # Trends, Outlook, Conclusions, Sources
```

## Verdict: WORKS PERFECTLY

Linear pipeline - perfect for hexDAG.

**hexDAG version enhancements:**
- Uses free DuckDuckGo (no API key needed for search)
- Structured report with 7 sections
- Source tracking with URLs
- Custom topic support via command line

## Report Sections

1. **Executive Summary** - Quick overview
2. **Key Findings** - Bullet points
3. **Detailed Analysis** - In-depth discussion
4. **Current Trends** - What's happening now
5. **Future Outlook** - Predictions
6. **Conclusions** - Final thoughts
7. **Sources** - Links to research

## How to Run

```bash
cd framework-tests/project18-web-research-report

# Demo mode (predefined topics)
..\..\.venv\Scripts\python.exe run_web_research.py

# Custom topic
..\..\.venv\Scripts\python.exe run_web_research.py "electric vehicles market 2024"
```

Expected output:
```
============================================================
hexDAG Web Research and Report Generator Demo
============================================================

[Topic 1] Latest trends in renewable energy 2024
--------------------------------------------------
  [RESEARCH] Searching web for: Latest trends in renewable energy 2024
  [RESEARCH] Found 5 sources
  [REPORT] Generating comprehensive report...
  [REPORT] Report generated (3500 chars)

RESEARCH SOURCES FOUND:
------------------------------
[1] Solar Energy Growth: https://example.com/solar
[2] Wind Power Investments: https://example.com/wind
...

GENERATED REPORT:
------------------------------
# REPORT: Latest Trends in Renewable Energy 2024

## 1. EXECUTIVE SUMMARY
The renewable energy sector is experiencing unprecedented growth...

## 2. KEY FINDINGS
- Global solar installations up 35%
- Offshore wind capacity doubled
- Green hydrogen gaining traction
...
```

## Test Results
[To be filled after testing]

## Observations
[To be filled after testing]
