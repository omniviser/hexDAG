# Project 1: Basic Chatbot - hexDAG Port

## Overview
This folder contains the hexDAG port of LangGraph's `project-01-basic-chatbot`.

## Files
- `chatbot_pipeline.yaml` - hexDAG YAML pipeline definition
- `run_chatbot.py` - Python runner script
- `README.md` - This file

## Original LangGraph Version
Located at: `reference_examples/langgraph-tutorials/project-01-basic-chatbot/main.py`

## How to Run

### hexDAG Version:
```bash
cd framework-tests/project1-chatbot
python run_chatbot.py
```

### Original LangGraph Version:
```bash
cd reference_examples/langgraph-tutorials/project-01-basic-chatbot
python main.py
```

## Key Differences

| Aspect        | LangGraph                 | hexDAG                       |
| ------------- | ------------------------- | ---------------------------- |
| Graph Type    | `StateGraph`              | `DirectedGraph`              |
| Node Addition | `add_node()`              | `graph.add(NodeSpec())`      |
| State         | `TypedDict` with reducers | Plain dict                   |
| Execution     | `app.invoke()` sync       | `orchestrator.run()` async   |
| LLM           | `ChatGoogleGenerativeAI`  | `google.generativeai` direct |

## Test Results
[To be filled after testing]

### Test Questions:
1. "What is the capital of France?"
2. "Tell me a fun fact about space."
3. "What is 25 multiplied by 4?"

### LangGraph Responses:
- Q1: [Fill in]
- Q2: [Fill in]
- Q3: [Fill in]

### hexDAG Responses:
- Q1: [Fill in]
- Q2: [Fill in]
- Q3: [Fill in]

## Observations
[To be filled after testing]
