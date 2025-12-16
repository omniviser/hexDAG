#!/usr/bin/env python
"""
hexDAG Blog Writing and Publishing Agent Demo
Ported from LangGraph project-22-blog-writing-and-publishing-agent

Pattern: Multi-Agent Content Pipeline
- Research topic from URLs (or use sample content)
- Write SEO-optimized blog post
- Proofread and refine
- Publish to WordPress (or save locally)

Run with: ..\..\venv\Scripts\python.exe run_blog_writer.py
"""
import asyncio
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load environment variables
env_path = project_root / "reference_examples" / "langgraph-tutorials" / ".env"
load_dotenv(env_path)

import google.generativeai as genai

# Configure Google Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY not found in .env file")
    sys.exit(1)

genai.configure(api_key=GEMINI_API_KEY)

# Try to import optional dependencies
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from wordpress_xmlrpc import Client, WordPressPost
    from wordpress_xmlrpc.methods.posts import NewPost
    WORDPRESS_AVAILABLE = True
except ImportError:
    WORDPRESS_AVAILABLE = False

try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False


# Organization info (from LangGraph config.py)
ORGANIZATION_NAME = "HERE AND NOW AI"

ORGANIZATION_DESCRIPTION = """
HERE AND NOW AI is India's leading autonomous Artificial Intelligence Research Institute,
dedicated to transforming the way AI is taught, researched, and applied. Founded in 2018,
we deliver AI education through practical programs for students, conduct research on
cutting-edge LLMs, RAG frameworks, and autonomous agents, and build intelligent systems
for enterprises and academic institutions.

Our vision is to build a generation of AI-native graduates empowered with real-world skills.
We offer courses in Business Analytics with AI and Full-Stack AI Developer programs.
"""

# Sample research content for demo
SAMPLE_RESEARCH_CONTENT = """
--- Content from https://www.anthropic.com/news/model-context-protocol ---

Model Context Protocol (MCP): A New Standard for AI Integration

Anthropic has introduced the Model Context Protocol (MCP), an open standard that enables
seamless integration between AI assistants and the systems where data lives. MCP provides
a universal protocol for connecting AI models to different data sources and tools.

Key Features:
- Open-source protocol for AI-data connectivity
- Standardized way to expose data and capabilities to AI models
- Pre-built integrations with popular enterprise systems
- Local-first architecture for security and privacy
- Support for both local and remote data sources

Why MCP Matters:
1. Reduces fragmentation in AI integrations
2. Makes it easier to build AI-powered applications
3. Enables AI assistants to access real-time data
4. Provides secure, standardized data access
5. Supports multiple AI providers and platforms

Technical Overview:
MCP uses a client-server architecture where:
- MCP Hosts (like Claude Desktop) connect to MCP Servers
- MCP Servers expose data and tools through the protocol
- Communication happens via JSON-RPC over stdin/stdout or HTTP

Getting Started:
Developers can start building with MCP using the official SDKs:
- Python SDK: pip install mcp
- TypeScript SDK: npm install @anthropic-ai/mcp

The protocol supports:
- Resource exposure (files, databases, APIs)
- Tool definitions (functions the AI can call)
- Prompt templates (reusable conversation starters)
- Sampling (controlled AI generation)

Enterprise Benefits:
- Unified integration layer for all AI applications
- Reduced development time for AI features
- Better security through standardized access patterns
- Easier maintenance and updates
- Support for compliance requirements
"""


async def research_agent(inputs: dict) -> dict:
    """
    Research content from URLs or use sample content.

    LangGraph version:
        tool = TavilySearch(api_key=tavily_api_key)
        results = tool.invoke(f"get_contents:{url}")

    hexDAG version: Uses requests or sample content.
    """
    urls = inputs.get("urls", [])
    use_sample = inputs.get("use_sample", True)

    print(f"  [RESEARCH] Starting research phase...")

    research_content = ""

    if use_sample or not urls:
        # Use sample content for demo
        research_content = SAMPLE_RESEARCH_CONTENT
        print(f"  [RESEARCH] Using sample research content (MCP article)")
    elif REQUESTS_AVAILABLE:
        # Fetch from URLs
        for url in urls:
            try:
                print(f"  [RESEARCH] Fetching: {url}")
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    # Basic text extraction (real implementation would parse HTML)
                    content = response.text[:5000]  # Limit content
                    research_content += f"\n--- Content from {url} ---\n{content}\n"
                    print(f"  [RESEARCH] Successfully fetched {len(content)} chars")
            except Exception as e:
                print(f"  [RESEARCH] Error fetching {url}: {e}")
    else:
        print(f"  [RESEARCH] requests library not available, using sample")
        research_content = SAMPLE_RESEARCH_CONTENT

    print(f"  [RESEARCH] Research complete: {len(research_content)} chars")

    return {
        "research_content": research_content,
        "organization_name": inputs.get("organization_name", ORGANIZATION_NAME),
        "organization_description": inputs.get("organization_description", ORGANIZATION_DESCRIPTION)
    }


async def writing_agent(inputs: dict) -> dict:
    """
    Write an SEO-optimized blog post based on research.

    LangGraph version:
        llm = ChatGoogleGenerativeAI(model=model, temperature=0.7)
        response = llm.invoke(prompt)
    """
    research_content = inputs.get("research_content", "")
    org_name = inputs.get("organization_name", ORGANIZATION_NAME)
    org_desc = inputs.get("organization_description", ORGANIZATION_DESCRIPTION)

    print(f"  [WRITE] Generating blog post...")

    prompt = f"""You are an expert SEO content writer for {org_name}.
Company Description: {org_desc}

Your task is to write a comprehensive, engaging, and SEO-optimized blog post based on the following research content:

{research_content}

Please adhere to the latest SEO best practices for 2025, including:
- Use the main keyword phrase naturally throughout the article
- Include related LSI keywords
- Write a compelling meta title and meta description at the top
- Use headings (H1, H2, H3) to structure the content with markdown syntax
- Write an introduction that hooks the reader and a conclusion that summarizes key points
- Ensure the article is at least 800 words long
- Include a call-to-action relevant to {org_name}

The tone should be professional, informative, and engaging.
Write in markdown format."""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    blog_post = response.text.strip()
    word_count = len(blog_post.split())

    print(f"  [WRITE] Blog post generated: {word_count} words")

    return {
        "blog_post": blog_post,
        "word_count": word_count
    }


async def proofreading_agent(inputs: dict) -> dict:
    """
    Proofread and refine the blog post.

    LangGraph version:
        llm = ChatGoogleGenerativeAI(model=model, temperature=0)
        response = llm.invoke(prompt)
    """
    blog_post = inputs.get("blog_post", "")

    print(f"  [PROOFREAD] Reviewing and polishing...")

    prompt = f"""You are an expert proofreader and editor.
Your task is to review the following blog post for any grammatical errors, spelling mistakes, or awkward phrasing.
Please also ensure the article is clear, concise, and easy to read.

Blog Post:
---
{blog_post}
---

Return the polished, final version of the blog post as clean markdown text.
Do not include any HTML tags, code fences, or meta-commentary.
Just return the improved article."""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    final_post = response.text.strip()

    # Clean up any code fences
    if final_post.startswith("```"):
        lines = final_post.split("\n")
        final_post = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    print(f"  [PROOFREAD] Proofreading complete")

    return {
        "final_blog_post": final_post
    }


async def publish_agent(inputs: dict) -> dict:
    """
    Publish to WordPress or save locally.

    LangGraph version:
        client = Client(wp_url, wp_username, wp_password)
        client.call(NewPost(post))

    hexDAG version: Supports both WordPress and local file saving.
    """
    final_post = inputs.get("final_blog_post", "")
    publish_mode = inputs.get("publish_mode", "local")  # "wordpress" or "local"

    print(f"  [PUBLISH] Publishing blog post...")

    # Extract title from the post
    lines = final_post.split('\n')
    post_title = "Automated Blog Post"
    post_content = final_post

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('# '):
            post_title = stripped.lstrip('# ').strip()
            post_content = '\n'.join(lines[i+1:]).strip()
            break

    result = {
        "title": post_title,
        "published": False,
        "publish_mode": publish_mode
    }

    if publish_mode == "wordpress" and WORDPRESS_AVAILABLE:
        # Publish to WordPress
        wp_url = os.getenv("WORDPRESS_URL")
        wp_username = os.getenv("WORDPRESS_USERNAME")
        wp_password = os.getenv("WORDPRESS_PASSWORD")

        if not all([wp_url, wp_username, wp_password]):
            print(f"  [PUBLISH] WordPress credentials not found in .env")
            print(f"  [PUBLISH] Falling back to local save")
            publish_mode = "local"
        else:
            try:
                # Convert markdown to HTML
                if MARKDOWN_AVAILABLE:
                    html_content = markdown.markdown(post_content)
                else:
                    html_content = post_content

                client = Client(wp_url, wp_username, wp_password)
                post = WordPressPost()
                post.title = post_title
                post.content = html_content
                post.post_status = 'publish'

                post_id = client.call(NewPost(post))
                print(f"  [PUBLISH] Successfully published to WordPress (ID: {post_id})")
                result["published"] = True
                result["post_id"] = post_id
                return result
            except Exception as e:
                print(f"  [PUBLISH] WordPress error: {e}")
                print(f"  [PUBLISH] Falling back to local save")
                publish_mode = "local"

    # Local file save
    if publish_mode == "local":
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)

        # Create filename from title
        safe_title = re.sub(r'[^\w\s-]', '', post_title)[:50].strip()
        safe_title = re.sub(r'[-\s]+', '-', safe_title).lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{safe_title}.md"

        output_path = output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# {post_title}\n\n")
            f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            f.write("---\n\n")
            f.write(post_content)

        print(f"  [PUBLISH] Saved to: {output_path}")
        result["published"] = True
        result["file_path"] = str(output_path)

    return result


async def run_blog_demo():
    """
    Demonstrate the full blog writing pipeline.
    """
    print("=" * 60)
    print("hexDAG Blog Writing Agent Demo")
    print("=" * 60)
    print()
    print("This agent creates blog posts through a multi-step pipeline:")
    print("  1. RESEARCH - Gather information from sources")
    print("  2. WRITE - Create SEO-optimized content")
    print("  3. PROOFREAD - Polish and refine")
    print("  4. PUBLISH - Save or publish to WordPress")
    print()
    print(f"Organization: {ORGANIZATION_NAME}")
    print()
    print("-" * 60)

    # Step 1: Research
    print("\n[Step 1/4] RESEARCH PHASE")
    print("-" * 40)
    research_result = await research_agent({
        "urls": ["https://www.anthropic.com/news/model-context-protocol"],
        "use_sample": True,  # Use sample for demo
        "organization_name": ORGANIZATION_NAME,
        "organization_description": ORGANIZATION_DESCRIPTION
    })

    # Step 2: Write
    print("\n[Step 2/4] WRITING PHASE")
    print("-" * 40)
    write_result = await writing_agent({
        "research_content": research_result["research_content"],
        "organization_name": research_result["organization_name"],
        "organization_description": research_result["organization_description"]
    })

    # Step 3: Proofread
    print("\n[Step 3/4] PROOFREADING PHASE")
    print("-" * 40)
    proofread_result = await proofreading_agent({
        "blog_post": write_result["blog_post"]
    })

    # Step 4: Publish
    print("\n[Step 4/4] PUBLISHING PHASE")
    print("-" * 40)
    publish_result = await publish_agent({
        "final_blog_post": proofread_result["final_blog_post"],
        "publish_mode": "local"  # Use local for demo
    })

    # Summary
    print("\n" + "=" * 60)
    print("BLOG WRITING COMPLETE")
    print("=" * 60)
    print(f"Title: {publish_result['title']}")
    print(f"Word Count: {write_result['word_count']}")
    print(f"Published: {'Yes' if publish_result['published'] else 'No'}")
    if 'file_path' in publish_result:
        print(f"Saved to: {publish_result['file_path']}")
    print()

    # Show preview
    print("BLOG POST PREVIEW (first 500 chars):")
    print("-" * 40)
    preview = proofread_result["final_blog_post"][:500]
    if len(proofread_result["final_blog_post"]) > 500:
        preview += "\n... (truncated)"
    print(preview)
    print()

    return publish_result


async def main():
    """Main entry point."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--wordpress":
        # Try to publish to WordPress
        print("WordPress publishing mode enabled")
        print("Note: Requires WORDPRESS_URL, WORDPRESS_USERNAME, WORDPRESS_PASSWORD in .env")

        research = await research_agent({"use_sample": True})
        write = await writing_agent(research)
        proofread = await proofreading_agent(write)
        result = await publish_agent({
            "final_blog_post": proofread["final_blog_post"],
            "publish_mode": "wordpress"
        })
        print(f"\nResult: {result}")
    else:
        # Run demo with local save
        await run_blog_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
