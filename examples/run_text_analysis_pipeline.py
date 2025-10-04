"""Run the text analysis pipeline from YAML manifest.

This example demonstrates how to:
1. Load a YAML pipeline manifest
2. Define custom functions for nodes
3. Execute the pipeline with the Orchestrator
4. Display results
"""

import asyncio
from pathlib import Path

from hexai.agent_factory.yaml_builder import YamlPipelineBuilder
from hexai.core.application.orchestrator import Orchestrator
from hexai.core.bootstrap import ensure_bootstrapped

# Ensure registry is bootstrapped
ensure_bootstrapped()

# Import transformers for NLP tasks
try:
    from transformers import pipeline as hf_pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  transformers not available. Install with: pip install transformers torch")


# Define the functions that the YAML references
def extract_keywords_fn(article_text: str) -> dict:
    """Extract keywords using NER."""
    if not TRANSFORMERS_AVAILABLE:
        # Fallback to simple word frequency
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
        }
        words = article_text.lower().split()
        word_freq = {}
        for word in words:
            cleaned = word.strip(".,!?;:")
            if cleaned and cleaned not in stop_words and len(cleaned) > 3:
                word_freq[cleaned] = word_freq.get(cleaned, 0) + 1
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, _ in sorted_words[:10]]
        return {"keywords": keywords, "keyword_count": len(keywords)}

    ner = hf_pipeline(
        "ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True
    )
    entities = ner(article_text[:512])
    keywords = []
    seen = set()
    for entity in entities:
        word = entity["word"].strip()
        if word.lower() not in seen:
            keywords.append(word)
            seen.add(word.lower())
        if len(keywords) >= 10:
            break

    return {"keywords": keywords, "keyword_count": len(keywords), "entities_found": len(entities)}


def analyze_sentiment_fn(article_text: str) -> dict:
    """Analyze sentiment using DistilBERT."""
    if not TRANSFORMERS_AVAILABLE:
        # Simple fallback
        positive_words = {"good", "great", "excellent", "happy", "love", "amazing"}
        negative_words = {"bad", "terrible", "hate", "awful", "poor", "horrible"}
        text_lower = article_text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        if pos_count > neg_count:
            return {"sentiment": "positive", "score": 0.8}
        if neg_count > pos_count:
            return {"sentiment": "negative", "score": 0.8}
        return {"sentiment": "neutral", "score": 0.5}

    sentiment = hf_pipeline(
        "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    result = sentiment(article_text[:512])[0]
    return {
        "sentiment": result["label"].lower(),
        "score": result["score"],
        "text_length": len(article_text),
    }


def summarize_text_fn(article_text: str) -> dict:
    """Summarize text using BART."""
    if not TRANSFORMERS_AVAILABLE:
        # Simple fallback - first 3 sentences
        sentences = article_text.split(". ")[:3]
        summary = ". ".join(sentences)
        if len(summary) > 200:
            summary = summary[:200] + "..."
        return {
            "summary": summary,
            "original_length": len(article_text),
            "summary_length": len(summary),
        }

    summarizer = hf_pipeline("summarization", model="facebook/bart-large-cnn")
    result = summarizer(article_text[:1024], max_length=200, min_length=30, do_sample=False)[0]
    summary = result["summary_text"]
    return {
        "summary": summary,
        "original_length": len(article_text),
        "summary_length": len(summary),
        "compression_ratio": len(summary) / len(article_text),
    }


def create_analysis_report(
    extract_keywords: dict, analyze_sentiment: dict, summarize_text: dict
) -> dict:
    """Combine all analysis results into a final report."""
    return {
        "report": {
            "keywords": extract_keywords.get("keywords", []),
            "keyword_count": extract_keywords.get("keyword_count", 0),
            "sentiment": analyze_sentiment.get("sentiment", "unknown"),
            "sentiment_score": analyze_sentiment.get("score", 0.0),
            "summary": summarize_text.get("summary", ""),
            "compression_ratio": summarize_text.get("compression_ratio", 0.0),
        }
    }


async def main():
    """Execute the text analysis pipeline."""
    # Sample article text
    sample_article = """
    Artificial intelligence has revolutionized the technology industry in recent years.
    Machine learning algorithms now power everything from smartphone assistants to
    autonomous vehicles. Companies like Google, Microsoft, and OpenAI are leading
    the charge in developing more sophisticated AI systems. The recent breakthroughs
    in large language models have been particularly impressive, enabling computers
    to understand and generate human-like text. However, concerns about AI safety
    and ethical implications continue to grow. Researchers emphasize the importance
    of responsible AI development to ensure these powerful technologies benefit humanity.
    Despite challenges, the future of AI looks promising with potential applications
    in healthcare, education, and scientific research.
    """

    # Load the YAML pipeline
    yaml_path = Path("examples/manifests/text-analysis-pipeline.yaml")
    builder = YamlPipelineBuilder()

    print(f"\n📄 Loading pipeline from: {yaml_path}")
    with open(yaml_path) as f:
        graph, metadata = builder.build_from_yaml_string(f.read())

    print(f"✓ Loaded pipeline: {metadata['name']}")
    print(f"  Description: {metadata.get('description', 'N/A')}")
    print(f"  Nodes: {len(graph.nodes)}")

    # Prepare context with input and function definitions
    context = {
        "input": {"article_text": sample_article.strip()},
        # Make functions available in context
        "extract_keywords_fn": extract_keywords_fn,
        "analyze_sentiment_fn": analyze_sentiment_fn,
        "summarize_text_fn": summarize_text_fn,
        "create_analysis_report": create_analysis_report,
    }

    # Execute the pipeline
    print("\n🚀 Executing pipeline...")
    if TRANSFORMERS_AVAILABLE:
        print("   (Loading models... this may take a minute on first run)")

    orchestrator = Orchestrator()
    results = await orchestrator.aexecute(graph, context=context)

    # Display results
    print("\n" + "=" * 80)
    print("📊 TEXT ANALYSIS RESULTS")
    print("=" * 80)

    report = results["generate_report"]["report"]

    print("\n🔑 Keywords:")
    print(f"   {', '.join(report['keywords'])}")
    print(f"   (Found {report['keyword_count']} keywords)")

    print("\n😊 Sentiment:")
    print(f"   {report['sentiment'].upper()} (confidence: {report['sentiment_score']:.1%})")

    print("\n📝 Summary:")
    print(f"   {report['summary']}")
    if "compression_ratio" in report:
        print(f"   (Compression: {report['compression_ratio']:.1%})")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
