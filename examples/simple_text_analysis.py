"""Simple text analysis example using transformers.

This example demonstrates:
1. Creating a DirectedGraph programmatically (no YAML)
2. Using transformers library for real NLP tasks
3. Parallel execution of independent nodes
4. Combining results

Run with:
    PYTHONPATH=. uv run python examples/simple_text_analysis.py

This example runs with DEBUG logging by default to show internal processing.
To reduce verbosity:
    HEXDAG_LOG_LEVEL=INFO PYTHONPATH=. uv run python examples/simple_text_analysis.py
To hide all logs:
    PYTHONPATH=. uv run python examples/simple_text_analysis.py 2>/dev/null
"""

import asyncio
import os

# Set log level to DEBUG for this example
# This must be done BEFORE importing hexai modules
os.environ.setdefault("HEXDAG_LOG_LEVEL", "DEBUG")

from hexdag.core.bootstrap import ensure_bootstrapped
from hexdag.core.domain.dag import DirectedGraph, NodeSpec
from hexdag.core.orchestration.orchestrator import Orchestrator

# Ensure registry is loaded
ensure_bootstrapped()

# Import transformers
try:
    from transformers import pipeline as hf_pipeline

    TRANSFORMERS_AVAILABLE = True
    print("✓ Transformers library available")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers not available - using fallback implementations")
    print("   Install with: pip install transformers torch sentencepiece")


# Define analysis functions
def extract_keywords(input_data: dict) -> dict:
    """Extract keywords using NER or word frequency."""
    text = input_data.get("text", "")
    max_keywords = 10

    # Try transformers first, fall back to simple method
    if TRANSFORMERS_AVAILABLE:
        try:
            ner = hf_pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                grouped_entities=True,
            )
            entities = ner(text[:512])
            keywords = list({entity["word"].strip() for entity in entities})[:max_keywords]
            return {"keywords": keywords, "method": "ner", "entities_found": len(entities)}
        except Exception:
            # Fall through to simple method if transformers fails
            pass

    # Simple fallback
    stop_words = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "is",
        "was",
    }
    words = text.lower().split()
    word_freq = {}
    for word in words:
        cleaned = word.strip(".,!?;:")
        if cleaned and cleaned not in stop_words and len(cleaned) > 3:
            word_freq[cleaned] = word_freq.get(cleaned, 0) + 1
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    keywords = [word for word, _ in sorted_words[:max_keywords]]
    return {"keywords": keywords, "method": "frequency"}


def analyze_sentiment(input_data: dict) -> dict:
    """Analyze sentiment using DistilBERT."""
    text = input_data.get("text", "")

    # Try transformers first, fall back to simple method
    if TRANSFORMERS_AVAILABLE:
        try:
            sentiment_pipeline = hf_pipeline(
                "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            result = sentiment_pipeline(text[:512])[0]
            return {
                "sentiment": result["label"].lower(),
                "score": result["score"],
                "method": "distilbert",
            }
        except Exception:
            # Fall through to simple method if transformers fails
            pass

    # Simple fallback
    positive_words = {"good", "great", "excellent", "happy", "love", "amazing", "wonderful"}
    negative_words = {"bad", "terrible", "hate", "awful", "poor", "horrible", "disappointing"}
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    if pos_count > neg_count:
        sentiment = "positive"
        score = min(0.9, 0.5 + (pos_count - neg_count) * 0.1)
    elif neg_count > pos_count:
        sentiment = "negative"
        score = min(0.9, 0.5 + (neg_count - pos_count) * 0.1)
    else:
        sentiment = "neutral"
        score = 0.5
    return {"sentiment": sentiment, "score": score, "method": "keyword"}


def summarize_text(input_data: dict) -> dict:
    """Summarize text using BART."""
    text = input_data.get("text", "")
    max_length = 100

    # Try transformers first, fall back to simple method
    if TRANSFORMERS_AVAILABLE:
        try:
            summarizer = hf_pipeline("summarization", model="facebook/bart-large-cnn")
            result = summarizer(text[:1024], max_length=max_length, min_length=30, do_sample=False)[
                0
            ]
            return {"summary": result["summary_text"], "method": "bart"}
        except Exception:
            # Fall through to simple method if transformers fails
            pass

    # Simple fallback - first few sentences
    sentences = [s.strip() for s in text.split(". ") if s.strip()][:3]
    summary = ". ".join(sentences)
    if not summary.endswith("."):
        summary += "."
    if len(summary) > max_length:
        summary = summary[:max_length] + "..."
    return {"summary": summary, "method": "extractive"}


def create_report(input_data: dict) -> dict:
    """Combine all analysis results."""
    # When a node has dependencies, it receives the combined results as input_data
    # with keys matching the node names
    keywords_result = input_data.get("keywords", {})
    sentiment_result = input_data.get("sentiment", {})
    summary_result = input_data.get("summary", {})

    return {
        "analysis_report": {
            "keywords": keywords_result.get("keywords", []),
            "sentiment": {
                "label": sentiment_result.get("sentiment", "unknown"),
                "confidence": sentiment_result.get("score", 0.0),
            },
            "summary": summary_result.get("summary", ""),
            "methods_used": {
                "keywords": keywords_result.get("method", "unknown"),
                "sentiment": sentiment_result.get("method", "unknown"),
                "summary": summary_result.get("method", "unknown"),
            },
        }
    }


async def main():
    """Run the text analysis pipeline."""
    # Sample text
    article = """
    Artificial intelligence is transforming the technology landscape at an unprecedented pace.
    Companies like Google, Microsoft, and OpenAI are pioneering breakthrough innovations in
    machine learning and natural language processing. The development of large language models
    has enabled remarkable capabilities in understanding and generating human-like text.
    These advancements promise to revolutionize industries from healthcare to education.
    However, experts emphasize the critical importance of responsible AI development to
    address concerns about safety, bias, and ethical implications. The future of AI holds
    tremendous potential for solving complex problems and enhancing human capabilities.
    """

    print("\n" + "=" * 80)
    print("TEXT ANALYSIS PIPELINE")
    print("=" * 80)
    print(f"\nArticle ({len(article)} chars):")
    print(article.strip())

    # Build the DAG
    graph = DirectedGraph()
    graph.name = "text_analysis_pipeline"  # Set pipeline name for logs

    # Add three independent analysis nodes (will run in parallel)
    # They all receive the same initial input (the article text)
    graph.add(NodeSpec(name="keywords", fn=extract_keywords, deps=frozenset()))

    graph.add(NodeSpec(name="sentiment", fn=analyze_sentiment, deps=frozenset()))

    graph.add(NodeSpec(name="summary", fn=summarize_text, deps=frozenset()))

    # Add report node that depends on all three
    graph.add(NodeSpec(name="report", fn=create_report).after("keywords", "sentiment", "summary"))

    print(f"\n📊 Pipeline: {len(graph.nodes)} nodes")
    print("   - keywords (parallel)")
    print("   - sentiment (parallel)")
    print("   - summary (parallel)")
    print("   - report (combines results)")

    # Execute the pipeline
    print("\n🚀 Executing pipeline...")
    if TRANSFORMERS_AVAILABLE:
        print("   Loading models... (this takes ~1 minute on first run)")

    orchestrator = Orchestrator()
    results = await orchestrator.run(graph, initial_input={"text": article.strip()})

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    report = results["report"]["analysis_report"]

    print("\n🔑 Keywords:")
    print(f"   {', '.join(report['keywords'])}")

    print("\n💭 Sentiment:")
    sentiment_emoji = (
        "😊"
        if report["sentiment"]["label"] == "positive"
        else "😞"
        if report["sentiment"]["label"] == "negative"
        else "😐"
    )  # noqa: E501
    print(f"   {sentiment_emoji} {report['sentiment']['label'].upper()}")
    print(f"   Confidence: {report['sentiment']['confidence']:.1%}")

    print("\n📝 Summary:")
    print(f"   {report['summary']}")

    print("\n🔧 Methods:")
    print(f"   Keywords: {report['methods_used']['keywords']}")
    print(f"   Sentiment: {report['methods_used']['sentiment']}")
    print(f"   Summary: {report['methods_used']['summary']}")

    print("\n" + "=" * 80)
    print("✓ Pipeline completed successfully")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
