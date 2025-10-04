"""Example usage of the text processing plugin.

This example demonstrates how to use the text processing nodes
programmatically without YAML manifests.
"""

from hexai.core.bootstrap import bootstrap_registry
from hexai.core.domain.dag import DirectedGraph
from hexai.core.orchestration import Orchestrator

# Bootstrap the registry to load plugins
bootstrap_registry()


async def main():
    """Run example text processing pipeline."""
    # Sample text to analyze
    sample_text = """
    The new smartphone features an amazing camera and excellent battery life.
    Users are extremely happy with the performance and build quality.
    However, some customers reported that the price is quite high.
    Overall, it's a great device that exceeded expectations.
    """

    # Create a DirectedGraph manually
    graph = DirectedGraph()

    # Add sentiment analysis node
    from hexai.core.registry import registry

    sentiment_factory = registry.get_node("sentiment_analyzer_node", namespace="textproc")
    sentiment_node = sentiment_factory(
        name="sentiment",
        text=sample_text,
        model="distilbert",
        deps=[],
    )
    graph.add_node(sentiment_node)

    # Add keyword extraction node
    keyword_factory = registry.get_node("keyword_extractor_node", namespace="textproc")
    keyword_node = keyword_factory(
        name="keywords",
        text=sample_text,
        max_keywords=5,
        deps=[],
    )
    graph.add_node(keyword_node)

    # Add summarization node
    summary_factory = registry.get_node("text_summarizer_node", namespace="textproc")
    summary_node = summary_factory(
        name="summary",
        text=sample_text,
        max_length=50,
        deps=[],
    )
    graph.add_node(summary_node)

    # Run the pipeline
    orchestrator = Orchestrator()
    results = await orchestrator.aexecute(graph, context={})

    # Print results
    print("\n=== Text Analysis Results ===\n")

    print("📊 Sentiment Analysis:")
    sentiment = results["sentiment"]
    print(f"  Sentiment: {sentiment['sentiment']}")
    print(f"  Confidence: {sentiment['score']:.2%}")
    print(f"  Text length: {sentiment['text_length']} chars\n")

    print("🔑 Keywords:")
    keywords = results["keywords"]
    print(f"  Keywords: {', '.join(keywords['keywords'])}")
    print(f"  Entities found: {keywords['entities_found']}")
    print(f"  Total keywords: {keywords['keyword_count']}\n")

    print("📝 Summary:")
    summary = results["summary"]
    print(f"  {summary['summary']}")
    print(f"  Compression: {summary['compression_ratio']:.1%}")
    print(f"  ({summary['summary_length']} chars from {summary['original_length']} chars)")


if __name__ == "__main__":
    import asyncio

    print("Loading models (this may take a minute on first run)...")
    asyncio.run(main())
