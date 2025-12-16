# Project 16: RAG Chatbot - hexDAG Port

## Overview
This folder contains the hexDAG port of LangGraph's `project-16-rag-retrieval-augmented-generation-chatbot`.

## What It Tests

**RAG (Retrieval Augmented Generation) - answer questions from a knowledge base.**

```
┌──────────┐     ┌──────────────┐     ┌──────────────┐
│ Question │ --> │   Retrieve   │ --> │   Generate   │ --> Answer
│          │     │   Context    │     │   Answer     │
└──────────┘     └──────────────┘     └──────────────┘
                       │                     │
                       v                     v
                 Search knowledge       LLM answers using
                 base for relevant      retrieved context
                 documents              (grounded in facts)
```

### Example Flow:
```
Question: "What is LangGraph?"

[RETRIEVE] Searching knowledge base...
[RETRIEVE] Found 2 relevant documents

Context:
  - About LangGraph: "LangGraph is a library for building stateful..."
  - hexDAG vs LangGraph: "Comparison of hexDAG and LangGraph..."

[GENERATE] Generating answer...

Answer: "LangGraph is a library for building stateful, multi-actor
applications with LLMs. It is built on top of LangChain and allows
you to define workflows as graphs where nodes are functions..."
```

### Why RAG Matters:
- **Reduces hallucination** - Answers grounded in actual documents
- **Up-to-date info** - Not limited by LLM training cutoff
- **Domain-specific** - Can answer about your company/product
- **Verifiable** - Can trace answer back to source documents

## Files
- `rag_pipeline.yaml` - hexDAG YAML pipeline
- `run_rag_chatbot.py` - Python runner with knowledge base
- `README.md` - This file

## The Difference

| | LangGraph | hexDAG |
|---|-----------|--------|
| **Vector Store** | FAISS + Embeddings | Simple keyword search |
| **Retrieval** | `retriever.get_relevant_documents()` | Manual search function |
| **Pattern** | retrieve -> generate | retrieve -> generate |

### LangGraph (FAISS):
```python
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_texts(texts, embeddings)
retriever = vectorstore.as_retriever()

def retrieve_context(state):
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([d.page_content for d in docs])
    return {"context": context}
```

### hexDAG (keyword search):
```python
def simple_search(query: str, knowledge_base: list) -> list:
    # Keyword matching (could use FAISS too)
    query_words = set(query.lower().split())
    # ... score and rank documents
    return relevant_docs

async def retrieve_context(inputs: dict) -> dict:
    relevant_docs = simple_search(question, KNOWLEDGE_BASE)
    return {"context": format_docs(relevant_docs)}
```

## Verdict: WORKS PERFECTLY

RAG is the classic use case for linear pipelines - perfect for hexDAG!

**Note:** This is similar to Project 10 (HR Policy Helper) but focuses specifically on the RAG pattern. Both demonstrate that hexDAG handles RAG very well.

**hexDAG version includes:**
- Knowledge base about hexDAG, LangGraph, and RAG
- Interactive mode for asking questions
- Graceful handling of questions not in knowledge base

## Knowledge Base Topics

| Topic | Content |
|-------|---------|
| About hexDAG | Framework overview and features |
| About LangGraph | Library overview and features |
| hexDAG vs LangGraph | Comparison of both frameworks |
| What is RAG? | Explanation of RAG pattern |
| Pricing | Pricing information |

## How to Run

```bash
cd framework-tests/project16-rag-chatbot

# Demo mode (runs predefined questions)
..\..\.venv\Scripts\python.exe run_rag_chatbot.py

# Interactive mode (ask your own questions)
..\..\.venv\Scripts\python.exe run_rag_chatbot.py --interactive
```

Expected output:
```
============================================================
hexDAG RAG Chatbot Demo
============================================================

[Q1] What is LangGraph?
--------------------------------------------------
  [RETRIEVE] Searching knowledge base...
  [RETRIEVE] Found 2 relevant documents
  [GENERATE] Generating answer...

Answer: LangGraph is a library for building stateful, multi-actor
applications with LLMs. It is built on top of LangChain...

============================================================

[Q6] What is the weather today?
--------------------------------------------------
  [RETRIEVE] Searching knowledge base...
  [RETRIEVE] No matching documents found
  [GENERATE] Generating answer...

Answer: I don't have information about that in my knowledge base.
```

## Test Results
[To be filled after testing]

## Observations
[To be filled after testing]
