# Project 10: HR Policy Helper - hexDAG Port

## Overview
This folder contains the hexDAG port of LangGraph's `project-10-decision-making-agent`.

## What It Tests

**RAG (Retrieval Augmented Generation) pattern.**

```
┌──────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Question   │ --> │ Retrieve Docs   │ --> │ LLM + Context   │ --> Answer
│              │     │ (Vector Search) │     │ (Generate)      │
└──────────────┘     └─────────────────┘     └─────────────────┘

"How many vacation    Finds: VACATION      "You get 20 days
 days do I get?"      POLICY document       of PTO per year..."
```

### Why RAG Matters:
- **LLMs have knowledge cutoffs** - RAG provides current/private data
- **Reduces hallucination** - Answer based on actual documents
- **Company-specific knowledge** - Internal policies, docs, procedures

### Example Flow:
```
Employee: "What is the remote work policy?"

[RETRIEVE] Searching policies...
[RETRIEVE] Found: Remote Work Policy

[ASSISTANT] Generating answer...

Answer: "You may work remotely up to 3 days per week.
        Core hours are 10am-3pm. Home office stipend: $500..."
```

## Files
- `hr_pipeline.yaml` - hexDAG YAML pipeline
- `run_hr_helper.py` - Python runner with policy retrieval
- `README.md` - This file

## The Difference

| | LangGraph | hexDAG |
|---|-----------|--------|
| **Vector Store** | FAISS + Embeddings | Simple keyword search (demo) |
| **Retrieval** | Built-in retriever | Manual implementation |
| **Pattern** | Same RAG pattern | Same RAG pattern |

### LangGraph (FAISS vector store):
```python
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_texts(texts, embeddings)
retriever = vectorstore.as_retriever()

# Retrieval happens automatically
docs = retriever.invoke(question)
```

### hexDAG (manual retrieval):
```python
async def retrieve_policies(inputs: dict) -> dict:
    question = inputs.get("question", "")

    # Simple keyword search (or use FAISS/Pinecone/etc)
    relevant_docs = simple_keyword_search(question, HR_POLICIES)

    return {"retrieved_context": format_docs(relevant_docs)}
```

## Verdict: WORKS WELL

hexDAG handles RAG pattern naturally! The linear flow (retrieve -> generate) is a perfect fit for DAGs.

**What hexDAG doesn't have:**
- Built-in vector store integration
- Built-in embedding support

**But this is fine because:**
- Vector stores are separate libraries (FAISS, Pinecone, Chroma)
- Easy to integrate in the retrieval node
- hexDAG focuses on orchestration, not storage

**Real production setup would use:**
```python
# Can use same FAISS setup as LangGraph
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

async def retrieve_policies(inputs: dict) -> dict:
    # Use FAISS for real vector search
    docs = vectorstore.similarity_search(question, k=3)
    return {"retrieved_context": docs}
```

## How to Run

```bash
cd framework-tests/project10-hr-policy-helper

# Demo mode (runs predefined questions)
..\..\.venv\Scripts\python.exe run_hr_helper.py

# Interactive mode (ask your own questions)
..\..\.venv\Scripts\python.exe run_hr_helper.py --interactive
```

Expected output:
```
============================================================
hexDAG HR Policy Helper Demo
============================================================

[Question 1] How many vacation days do I get?
--------------------------------------------------
  [RETRIEVE] Searching policies for: How many vacation days do I get?
  [RETRIEVE] Found 2 relevant policies
  [ASSISTANT] Generating answer...

Answer: Full-time employees receive 20 days of paid time off (PTO) per year.
PTO accrues at 1.67 days per month...
```

## Test Results
[To be filled after testing]

## Observations
[To be filled after testing]
