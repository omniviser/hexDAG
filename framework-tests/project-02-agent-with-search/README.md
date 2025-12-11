# Project 2: Agent with Search Tool - hexDAG Port

## Overview
This folder contains the hexDAG port of LangGraph's `project-02-agent-with-search-tool`.

## Files
- `run_agent.py` - Python runner with search capability
- `README.md` - This file

## Original LangGraph Version
Located at: `reference_examples/langgraph-tutorials/project-02-agent-with-search-tool/tavily_search.py`

## Key Differences

| Aspect | LangGraph | hexDAG |
|--------|-----------|--------|
| Search API | Tavily (requires API key) | DuckDuckGo (free) |
| Tool Binding | `llm.bind_tools()` | Manual in prompt |
| Conditional | `add_conditional_edges()` | Logic in Python |
| Memory | `InMemorySaver` | Manual list |

## How to Run

### Install search dependency:
```bash
pip install duckduckgo-search
```

### hexDAG Version:
```bash
cd framework-tests/project2-agent-with-search
..\..\.venv\Scripts\python.exe run_agent.py
```

## Test Questions
1. "What is the current weather in Tokyo?"
2. "Who won the latest Nobel Prize in Physics?"
3. "What are the top news headlines today?"

## Test Results
[To be filled after testing]

## Observations
[To be filled after testing]
