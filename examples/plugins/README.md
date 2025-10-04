# HexDAG Example Plugins

This directory contains example plugins demonstrating how to extend HexDAG with custom components.

## Plugins

### 1. Text Processing Plugin (`text_processing/`)

Demonstrates custom nodes for text analysis and transformation.

**Components:**
- `sentiment_analyzer_node` - Analyze text sentiment
- `text_summarizer_node` - Summarize long text
- `keyword_extractor_node` - Extract keywords from text

**Usage:**
```python
# Install plugin
pip install -e examples/plugins/text_processing

# Use in YAML
kind: textproc:sentiment_analyzer_node
```

## Creating Your Own Plugin

1. **Create plugin structure:**
   ```
   my_plugin/
   ├── pyproject.toml
   ├── my_plugin/
   │   ├── __init__.py
   │   └── nodes.py
   ```

2. **Define custom nodes:**
   ```python
   from hexai.core.registry import node
   from hexai.core.domain.dag import NodeSpec

   @node(name="custom_node", namespace="myplugin")
   class CustomNode:
       def __call__(self, name: str, **params) -> NodeSpec:
           return NodeSpec(name, my_function, **params)
   ```

3. **Register in pyproject.toml:**
   ```toml
   [project.entry-points."hexai.plugins"]
   myplugin = "my_plugin"
   ```

4. **Use in pipelines:**
   ```yaml
   nodes:
     - kind: myplugin:custom_node
       metadata:
         name: my_node
       spec:
         # ... node params
   ```
