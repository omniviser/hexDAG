"""Custom node factories for text processing."""

from typing import Any

from hexai.core.domain.dag import NodeSpec
from hexai.core.registry import node
from hexai.core.registry.models import NodeSubtype

try:
    from transformers import pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@node(name="sentiment_analyzer_node", subtype=NodeSubtype.FUNCTION, namespace="textproc")
class SentimentAnalyzerNode:
    """Factory for creating sentiment analysis nodes."""

    def __call__(
        self,
        name: str,
        text: str,
        model: str = "simple",
        deps: list[str] | None = None,
        **kwargs: Any,
    ) -> NodeSpec:
        """Create a sentiment analysis node.

        Parameters
        ----------
        name : str
            Node name
        text : str
            Text to analyze (can use template variables)
        model : str, optional
            Sentiment model to use, by default "simple"
        deps : list[str] | None, optional
            Node dependencies
        **kwargs : Any
            Additional parameters

        Returns
        -------
        NodeSpec
            Node specification for sentiment analysis
        """

        def analyze_sentiment(text_input: str) -> dict[str, Any]:
            """Analyze text sentiment using transformers.

            Parameters
            ----------
            text_input : str
                Text to analyze

            Returns
            -------
            dict[str, Any]
                Sentiment analysis result
            """
            if not TRANSFORMERS_AVAILABLE:
                msg = "transformers library not installed. Install with: pip install transformers torch"
                raise ImportError(msg)

            # Use distilbert for sentiment analysis
            sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
            )

            result = sentiment_pipeline(text_input[:512])[0]  # Limit to 512 tokens

            return {
                "sentiment": result["label"].lower(),
                "score": result["score"],
                "model": model,
                "text_length": len(text_input),
            }

        # Store text template in params for runtime resolution
        kwargs["text_template"] = text
        kwargs["model"] = model

        return NodeSpec(
            name=name,
            fn=analyze_sentiment,
            deps=frozenset(deps or []),
            params=kwargs,
        )


@node(name="text_summarizer_node", subtype=NodeSubtype.FUNCTION, namespace="textproc")
class TextSummarizerNode:
    """Factory for creating text summarization nodes."""

    def __call__(
        self,
        name: str,
        text: str,
        max_length: int = 100,
        deps: list[str] | None = None,
        **kwargs: Any,
    ) -> NodeSpec:
        """Create a text summarization node.

        Parameters
        ----------
        name : str
            Node name
        text : str
            Text to summarize
        max_length : int, optional
            Maximum summary length, by default 100
        deps : list[str] | None, optional
            Node dependencies
        **kwargs : Any
            Additional parameters

        Returns
        -------
        NodeSpec
            Node specification for text summarization
        """

        def summarize_text(text_input: str) -> dict[str, Any]:
            """Summarize text using transformers.

            Parameters
            ----------
            text_input : str
                Text to summarize

            Returns
            -------
            dict[str, Any]
                Summarization result
            """
            if not TRANSFORMERS_AVAILABLE:
                msg = "transformers library not installed. Install with: pip install transformers torch"
                raise ImportError(msg)

            # Use BART for summarization
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

            # BART can handle up to 1024 tokens
            result = summarizer(
                text_input[:1024],
                max_length=max_length,
                min_length=30,
                do_sample=False,
            )[0]

            summary = result["summary_text"]

            return {
                "summary": summary,
                "original_length": len(text_input),
                "summary_length": len(summary),
                "compression_ratio": len(summary) / len(text_input) if text_input else 0,
            }

        kwargs["text_template"] = text
        kwargs["max_length"] = max_length

        return NodeSpec(
            name=name,
            fn=summarize_text,
            deps=frozenset(deps or []),
            params=kwargs,
        )


@node(name="keyword_extractor_node", subtype=NodeSubtype.FUNCTION, namespace="textproc")
class KeywordExtractorNode:
    """Factory for creating keyword extraction nodes."""

    def __call__(
        self,
        name: str,
        text: str,
        max_keywords: int = 5,
        deps: list[str] | None = None,
        **kwargs: Any,
    ) -> NodeSpec:
        """Create a keyword extraction node.

        Parameters
        ----------
        name : str
            Node name
        text : str
            Text to extract keywords from
        max_keywords : int, optional
            Maximum number of keywords, by default 5
        deps : list[str] | None, optional
            Node dependencies
        **kwargs : Any
            Additional parameters

        Returns
        -------
        NodeSpec
            Node specification for keyword extraction
        """

        def extract_keywords(text_input: str) -> dict[str, Any]:
            """Extract keywords from text using NER pipeline.

            Parameters
            ----------
            text_input : str
                Text to extract keywords from

            Returns
            -------
            dict[str, Any]
                Keyword extraction result
            """
            if not TRANSFORMERS_AVAILABLE:
                msg = "transformers library not installed. Install with: pip install transformers torch"
                raise ImportError(msg)

            # Use NER to extract entities as keywords
            ner_pipeline = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                grouped_entities=True,
            )

            entities = ner_pipeline(text_input[:512])  # Limit to 512 tokens

            # Extract unique entity words
            keywords = []
            seen = set()
            for entity in entities:
                word = entity["word"].strip()
                if word.lower() not in seen:
                    keywords.append(word)
                    seen.add(word.lower())
                if len(keywords) >= max_keywords:
                    break

            # If not enough named entities, fall back to simple frequency
            if len(keywords) < max_keywords:
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
                words = text_input.lower().split()
                word_freq = {}
                for word in words:
                    cleaned = word.strip(".,!?;:")
                    if (
                        cleaned
                        and cleaned not in stop_words
                        and len(cleaned) > 3
                        and cleaned not in seen
                    ):
                        word_freq[cleaned] = word_freq.get(cleaned, 0) + 1

                sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
                for word, _ in sorted_words:
                    if len(keywords) >= max_keywords:
                        break
                    keywords.append(word)

            return {
                "keywords": keywords[:max_keywords],
                "keyword_count": len(keywords[:max_keywords]),
                "entities_found": len(entities),
                "text_length": len(text_input),
            }

        kwargs["text_template"] = text
        kwargs["max_keywords"] = max_keywords

        return NodeSpec(
            name=name,
            fn=extract_keywords,
            deps=frozenset(deps or []),
            params=kwargs,
        )
