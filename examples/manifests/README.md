# HexDAG Example Manifests

This directory contains example YAML pipeline manifests demonstrating various features.

## Manifests

### text-analysis-pipeline.yaml

Complete text analysis workflow using custom plugin nodes with state-of-the-art NLP models.

**Features:**
- Uses `textproc` plugin nodes powered by Hugging Face Transformers:
  - **Sentiment Analysis**: DistilBERT (SST-2 fine-tuned)
  - **Text Summarization**: BART-large-cnn
  - **Keyword Extraction**: BERT-large NER
- Parallel execution of independent analysis tasks
- Combines results with function node
- Input schema validation

**Requirements:**
```bash
cd examples/plugins/text_processing
pip install -e .  # Installs transformers, torch, sentencepiece
```

**Note**: First run downloads pre-trained models (~500MB-1GB). Subsequent runs use cached models.

**Usage:**
```bash
# Validate the manifest
hexdag validate examples/manifests/text-analysis-pipeline.yaml

# View pipeline structure
hexdag schema explain textproc:keyword_extractor_node

# Run the example
cd examples/plugins/text_processing
python example_usage.py
```

### sentiment-classifier.yaml

Customer feedback classification pipeline.

**Features:**
- Input validation with function node
- Sentiment analysis with plugin node
- Conditional routing based on sentiment
- Different handlers for positive/negative feedback

**Usage:**
```bash
# Validate
hexdag validate examples/manifests/sentiment-classifier.yaml --explain

# Create similar pipeline
hexdag create from-schema sentiment_analyzer_node --namespace textproc
```

## Using These Manifests

### 1. Install Plugin (if using plugin nodes)

```bash
cd examples/plugins/text_processing
pip install -e .
```

### 2. Validate Manifest

```bash
hexdag validate examples/manifests/text-analysis-pipeline.yaml
```

### 3. Use in Python

```python
from hexai.agent_factory.yaml_builder import YamlPipelineBuilder

# Build pipeline from YAML
builder = YamlPipelineBuilder()
with open("examples/manifests/text-analysis-pipeline.yaml") as f:
    graph, metadata = builder.build_from_yaml_string(f.read())

# Execute pipeline
from hexai.core.application.orchestrator import Orchestrator

orchestrator = Orchestrator()
results = await orchestrator.run(
    graph,
    context={"input": {"article_text": "Your text here..."}}
)
```

## Creating Your Own Manifests

Use the CLI to bootstrap new manifests:

```bash
# Create minimal pipeline
hexdag create pipeline --name my-pipeline

# Create from specific node schema
hexdag create from-schema llm_node --output llm-pipeline.yaml

# Create full-featured template
hexdag create pipeline --template full --output production.yaml
```

## Best Practices

1. **Use Input Schemas**: Define expected input structure for validation
2. **Add Descriptions**: Document your pipeline and nodes
3. **Version Your Manifests**: Use semantic versioning in metadata
4. **Tag Appropriately**: Use tags for organization and discovery
5. **Validate Before Use**: Always run `hexdag validate` before deployment
