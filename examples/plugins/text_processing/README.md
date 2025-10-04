# Text Processing Plugin

Example HexDAG plugin for text analysis and transformation using state-of-the-art NLP models.

This plugin demonstrates how to integrate external ML libraries (Hugging Face Transformers) into HexDAG workflows.

## Features

- **Real NLP Models**: Uses Hugging Face Transformers for production-grade text processing
- **Sentiment Analysis**: DistilBERT model fine-tuned on SST-2 dataset
- **Text Summarization**: BART-large-cnn model for abstractive summarization
- **Keyword Extraction**: BERT-large NER model for entity-based keyword extraction

## Installation

```bash
cd examples/plugins/text_processing
pip install -e .
```

This will install the following dependencies:
- `transformers>=4.30.0` - Hugging Face Transformers library
- `torch>=2.0.0` - PyTorch deep learning framework
- `sentencepiece>=0.1.99` - Tokenization library

**Note**: First run will download pre-trained models (~500MB-1GB). Models are cached locally for subsequent runs.

## Nodes

### sentiment_analyzer_node

Analyze text sentiment using DistilBERT fine-tuned on SST-2.

**Model**: `distilbert-base-uncased-finetuned-sst-2-english`

**Parameters:**
- `text` (str): Text to analyze (truncated to 512 tokens)
- `model` (str, optional): Model identifier (default: "simple")

**Returns:**
- `sentiment`: "positive" or "negative"
- `score`: Confidence score (0-1)
- `model`: Model identifier used
- `text_length`: Input text length

**Example:**
```yaml
kind: textproc:sentiment_analyzer_node
metadata:
  name: sentiment
spec:
  text: "{{input.user_text}}"
  model: "distilbert"
```

### text_summarizer_node

Summarize long text using BART-large-cnn.

**Model**: `facebook/bart-large-cnn`

**Parameters:**
- `text` (str): Text to summarize (truncated to 1024 tokens)
- `max_length` (int, optional): Maximum summary length in tokens (default: 100)

**Returns:**
- `summary`: Generated summary text
- `original_length`: Input text length
- `summary_length`: Summary text length
- `compression_ratio`: Summary/original length ratio

**Example:**
```yaml
kind: textproc:text_summarizer_node
metadata:
  name: summarizer
spec:
  text: "{{input.document}}"
  max_length: 150
```

### keyword_extractor_node

Extract keywords using Named Entity Recognition (NER).

**Model**: `dbmdz/bert-large-cased-finetuned-conll03-english`

**Parameters:**
- `text` (str): Text to extract keywords from (truncated to 512 tokens)
- `max_keywords` (int, optional): Maximum number of keywords (default: 5)

**Returns:**
- `keywords`: List of extracted keywords
- `keyword_count`: Number of keywords returned
- `entities_found`: Number of named entities detected
- `text_length`: Input text length

**Strategy**:
1. Extract named entities using BERT NER
2. If not enough entities, fall back to frequency-based extraction

**Example:**
```yaml
kind: textproc:keyword_extractor_node
metadata:
  name: keywords
spec:
  text: "{{input.article}}"
  max_keywords: 10
```

## Performance Considerations

- **Model Loading**: Models are loaded on first use and kept in memory
- **Token Limits**:
  - Sentiment analysis: 512 tokens
  - Summarization: 1024 tokens
  - Keyword extraction: 512 tokens
- **Memory Usage**: Expect ~2-3GB RAM per model loaded

## Usage

### Running the Text Analysis Example

The easiest way to use this plugin is through the provided example script:

```bash
# From hexdag root directory (with DEBUG logging)
PYTHONPATH=. uv run python examples/simple_text_analysis.py

# For cleaner output (INFO level)
HEXDAG_LOG_LEVEL=INFO PYTHONPATH=. uv run python examples/simple_text_analysis.py

# No logs at all
PYTHONPATH=. uv run python examples/simple_text_analysis.py 2>/dev/null
```

**What it does:**
- Analyzes a sample article about AI
- Runs sentiment analysis, summarization, and keyword extraction in parallel
- Combines results into a comprehensive report
- Works with OR without transformers installed (graceful fallback)
- Runs with DEBUG logging by default to show internal processing

**With transformers installed:**
```bash
pip install transformers torch sentencepiece
PYTHONPATH=. uv run python examples/simple_text_analysis.py
```

Output will show:
```
🔧 Methods:
   Keywords: ner                    ← Uses BERT NER
   Sentiment: distilbert            ← Uses DistilBERT
   Summary: bart                    ← Uses BART
```

**Without transformers (fallback mode):**
```bash
# Run the example
PYTHONPATH=. uv run python examples/simple_text_analysis.py

# For cleaner output without debug logs:
PYTHONPATH=. uv run python examples/simple_text_analysis.py 2>/dev/null
```

Output will show:
```
🔧 Methods:
   Keywords: frequency              ← Simple word frequency
   Sentiment: keyword               ← Keyword matching
   Summary: extractive              ← Sentence extraction
```

### Using in Your Code

```python
from hexai.core.application.orchestrator import Orchestrator
from hexai.core.domain.dag import DirectedGraph, NodeSpec

def analyze_sentiment(input_data: dict) -> dict:
    """Your sentiment analysis function."""
    text = input_data.get("text", "")

    # Import transformers
    from transformers import pipeline
    sentiment = pipeline("sentiment-analysis",
                        model="distilbert-base-uncased-finetuned-sst-2-english")

    result = sentiment(text[:512])[0]
    return {
        "sentiment": result["label"].lower(),
        "score": result["score"]
    }

# Build and run pipeline
async def main():
    graph = DirectedGraph()
    graph.add(NodeSpec(name="sentiment", fn=analyze_sentiment))

    orchestrator = Orchestrator()
    results = await orchestrator.run(
        graph,
        initial_input={"text": "This is amazing!"}
    )

    print(results["sentiment"])
    # {'sentiment': 'positive', 'score': 0.9998}

import asyncio
asyncio.run(main())
```

## File Structure

```
text_processing/
├── README.md                      # This file
├── pyproject.toml                 # Package configuration with dependencies
├── text_processing_plugin/
│   ├── __init__.py               # Plugin initialization
│   └── nodes.py                  # Node implementations
└── example_usage.py              # Standalone usage example
```

## Advanced: Using as a Plugin

This plugin can be registered with hexDAG's plugin system:

1. **Install the plugin:**
   ```bash
   cd examples/plugins/text_processing
   pip install -e .
   ```

2. **Plugin is auto-discovered** via entry points in `pyproject.toml`:
   ```toml
   [project.entry-points."hexai.plugins"]
   textproc = "text_processing_plugin"
   ```

3. **List registered nodes:**
   ```bash
   uv run hexdag registry list --type node --namespace textproc
   ```

4. **Get node schema:**
   ```bash
   uv run hexdag schema get sentiment_analyzer_node --namespace textproc
   ```

## Troubleshooting

### "transformers not available"

This is expected! The code works without transformers using fallback implementations. Install transformers only if you want production-quality NLP:

```bash
pip install transformers torch sentencepiece
```

### Model download is slow

First run downloads 500MB-1GB of models. Subsequent runs use cached models from `~/.cache/huggingface/`.

### Out of memory

Models require ~2-3GB RAM. Reduce model usage or process text in smaller chunks:

```python
# Process in chunks
def analyze_in_chunks(text: str, chunk_size: int = 512):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    results = [analyze_sentiment({"text": chunk}) for chunk in chunks]
    return combine_results(results)
```

### Too much debug output

If you want cleaner output without debug logs:

```bash
# Suppress all logs (stderr)
PYTHONPATH=. uv run python examples/simple_text_analysis.py 2>/dev/null

# Or filter out debug lines
PYTHONPATH=. uv run python examples/simple_text_analysis.py 2>&1 | grep -v "DEBUG"
```

## See Also

- Complete working example: [examples/simple_text_analysis.py](../../simple_text_analysis.py)
- CLI usage: `uv run hexdag --help`
- Main documentation: [docs/README.md](../../../docs/README.md)
