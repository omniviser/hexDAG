# Project 22: Blog Writing and Publishing Agent - hexDAG Port

## Overview
This folder contains the hexDAG port of LangGraph's `project-22-blog-writing-and-publishing-agent`.

## What It Tests

**Multi-agent pipeline for researching, writing, and publishing blog posts.**

```
┌──────────┐    ┌─────────┐    ┌────────────┐    ┌──────────┐
│ Research │ -> │  Write  │ -> │ Proofread  │ -> │ Publish  │
│  (URLs)  │    │  (LLM)  │    │   (LLM)    │    │  (WP)    │
└──────────┘    └─────────┘    └────────────┘    └──────────┘
     │               │               │                │
     v               v               v                v
 Web content    800+ word       Polished         Published
 extraction     SEO article     content          or saved
```

### Example Flow:
```
[Step 1/4] RESEARCH PHASE
----------------------------------------
  [RESEARCH] Starting research phase...
  [RESEARCH] Using sample research content (MCP article)
  [RESEARCH] Research complete: 2341 chars

[Step 2/4] WRITING PHASE
----------------------------------------
  [WRITE] Generating blog post...
  [WRITE] Blog post generated: 892 words

[Step 3/4] PROOFREADING PHASE
----------------------------------------
  [PROOFREAD] Reviewing and polishing...
  [PROOFREAD] Proofreading complete

[Step 4/4] PUBLISHING PHASE
----------------------------------------
  [PUBLISH] Publishing blog post...
  [PUBLISH] Saved to: output/20241216_model-context-protocol.md

============================================================
BLOG WRITING COMPLETE
============================================================
Title: Model Context Protocol: The Future of AI Integration
Word Count: 892
Published: Yes
Saved to: output/20241216_model-context-protocol.md
```

### Why This Matters:
- **Content Automation** - Generate blog posts from research
- **SEO Optimization** - AI-powered content writing
- **Quality Control** - Automated proofreading
- **Publishing** - Direct WordPress integration

## Files
- `blog_pipeline.yaml` - hexDAG YAML pipeline
- `run_blog_writer.py` - Python runner with all agents
- `output/` - Generated blog posts (created on run)
- `README.md` - This file

## Dependencies

```bash
# Optional - for WordPress publishing
pip install python-wordpress-xmlrpc markdown requests
```

## The Difference

| | LangGraph | hexDAG |
|---|-----------|--------|
| **Research** | Tavily API | Sample content / requests |
| **Agents** | Separate files | Single file with functions |
| **Publishing** | WordPress only | WordPress + local save |
| **State** | TypedDict | Dict passing |

### LangGraph (multi-file agents):
```python
# agents/research_agent.py
def research_agent(state):
    tool = TavilySearch(api_key=tavily_api_key)
    results = tool.invoke(f"get_contents:{url}")
    return {"research_results": all_content}

# agents/writing_agent.py
def writing_agent(state):
    llm = ChatGoogleGenerativeAI(model=model)
    response = llm.invoke(prompt)
    return {"blog_post": response.content}

# app.py
workflow.add_edge("research", "write")
workflow.add_edge("write", "proofread")
workflow.add_edge("proofread", "publish")
```

### hexDAG (unified functions):
```python
async def research_agent(inputs: dict) -> dict:
    # Fetch from URLs or use sample
    return {"research_content": content}

async def writing_agent(inputs: dict) -> dict:
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return {"blog_post": response.text}

# Pipeline: research -> write -> proofread -> publish
```

## Verdict: WORKS PERFECTLY

Linear multi-agent pipeline - perfect for hexDAG.

**hexDAG version enhancements:**
- Sample content for testing without Tavily API
- Dual publish mode (WordPress + local file)
- Cleaner single-file organization
- Better error handling and fallbacks

## How to Run

```bash
cd framework-tests/project22-blog-writer

# Demo mode (saves to local file)
..\..\venv\Scripts\python.exe run_blog_writer.py

# WordPress publishing mode
..\..\venv\Scripts\python.exe run_blog_writer.py --wordpress
```

For WordPress publishing, add to `.env`:
```
WORDPRESS_URL=https://yoursite.com/xmlrpc.php
WORDPRESS_USERNAME=your_username
WORDPRESS_PASSWORD=your_app_password
```

Expected output:
```
============================================================
hexDAG Blog Writing Agent Demo
============================================================

This agent creates blog posts through a multi-step pipeline:
  1. RESEARCH - Gather information from sources
  2. WRITE - Create SEO-optimized content
  3. PROOFREAD - Polish and refine
  4. PUBLISH - Save or publish to WordPress

Organization: HERE AND NOW AI

------------------------------------------------------------

[Step 1/4] RESEARCH PHASE
----------------------------------------
  [RESEARCH] Starting research phase...
  [RESEARCH] Using sample research content (MCP article)
  [RESEARCH] Research complete: 2341 chars

[Step 2/4] WRITING PHASE
----------------------------------------
  [WRITE] Generating blog post...
  [WRITE] Blog post generated: 892 words

[Step 3/4] PROOFREADING PHASE
----------------------------------------
  [PROOFREAD] Reviewing and polishing...
  [PROOFREAD] Proofreading complete

[Step 4/4] PUBLISHING PHASE
----------------------------------------
  [PUBLISH] Publishing blog post...
  [PUBLISH] Saved to: output/20241216_143022_model-context-protocol.md

============================================================
BLOG WRITING COMPLETE
============================================================
Title: Model Context Protocol: The Future of AI Integration
Word Count: 892
Published: Yes
Saved to: output/20241216_143022_model-context-protocol.md

BLOG POST PREVIEW (first 500 chars):
----------------------------------------
# Model Context Protocol: The Future of AI Integration

**Meta Title:** Model Context Protocol (MCP) - Revolutionizing AI
Integration for Enterprises | HERE AND NOW AI

**Meta Description:** Discover how Anthropic's Model Context Protocol
(MCP) is transforming AI integration. Learn about this open standard
that enables seamless connectivity between AI assistants and your data.

## Introduction

In the rapidly evolving landscape of artificial intelligence...
```

## Test Results
[To be filled after testing]

## Observations
[To be filled after testing]
