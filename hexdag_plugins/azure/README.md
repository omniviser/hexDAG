# hexdag-azure

Azure OpenAI adapter plugin for the hexDAG framework.

## Overview

This plugin provides production-ready Azure OpenAI integration for hexDAG pipelines, supporting:

- **Azure-hosted OpenAI endpoints** with deployment-based model access
- **GPT-4, GPT-3.5-turbo, and fine-tuned models**
- **Native tool calling** support for agent workflows
- **Automatic secret resolution** from environment variables
- **Health checks** for connectivity monitoring
- **YAML-first configuration** for declarative pipelines

## Installation

```bash
# Install from source (development)
cd hexdag_plugins/azure
uv pip install -e .

# Or install from PyPI (when published)
pip install hexdag-azure
```

## Quick Start

### Environment Setup

Set your Azure OpenAI credentials:

```bash
export AZURE_OPENAI_API_KEY="your-api-key"
```

### YAML Pipeline

```yaml
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: azure-analysis-pipeline
spec:
  nodes:
    - kind: llm_node
      metadata:
        name: azure_analyzer
      spec:
        adapter:
          type: azure_openai
          params:
            resource_name: "my-openai-eastus"
            deployment_id: "gpt-4"
            api_version: "2024-02-15-preview"
            temperature: 0.7
        prompt_template: "Analyze the following data: {{input}}"
      dependencies: []
```

### Python API

```python
from hexdag_plugins.azure import AzureOpenAIAdapter
from hexdag.core.ports.llm import Message

# Create adapter (API key auto-resolved from AZURE_OPENAI_API_KEY)
adapter = AzureOpenAIAdapter(
    resource_name="my-openai-eastus",
    deployment_id="gpt-4",
    api_version="2024-02-15-preview",
    temperature=0.7,
)

# Generate response
messages = [Message(role="user", content="Hello, Azure!")]
response = await adapter.aresponse(messages)
print(response)
```

## Configuration

### Required Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `resource_name` | str | Azure OpenAI resource name | `"my-openai-eastus"` |
| `deployment_id` | str | Azure deployment name | `"gpt-4"` or `"gpt-35-turbo"` |
| `api_key` | str | Azure OpenAI API key | Auto-resolved from `AZURE_OPENAI_API_KEY` |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_version` | str | `"2024-02-15-preview"` | Azure OpenAI API version |
| `temperature` | float | `0.7` | Sampling temperature (0.0-2.0) |
| `max_tokens` | int | `None` | Maximum tokens in response |
| `timeout` | float | `30.0` | Request timeout in seconds |
| `embedding_deployment_id` | str | `None` | Azure deployment for embeddings (e.g., `"text-embedding-3-small"`) |
| `embedding_dimensions` | int | `None` | Embedding dimensionality (for text-embedding-3 models) |

## Features

### Unified LLM + Embedding Adapter

Azure OpenAI adapter implements both text generation and embedding capabilities in a single unified interface:

```python
# Single adapter for both LLM and embeddings
adapter = AzureOpenAIAdapter(
    resource_name="my-openai-eastus",
    deployment_id="gpt-4",  # For text generation
    embedding_deployment_id="text-embedding-3-small",  # For embeddings
    embedding_dimensions=1536,
)

# Use for text generation
response = await adapter.aresponse(messages)

# Use for embeddings
embedding = await adapter.aembed("Hello, world!")
embeddings = await adapter.aembed_batch(["Text 1", "Text 2", "Text 3"])
```

This unified approach simplifies API key management and resource configuration when using multiple Azure OpenAI capabilities.

### Tool Calling Support

Azure OpenAI adapter supports native tool/function calling:

```yaml
nodes:
  - kind: agent_node
    metadata:
      name: azure_agent
    spec:
      llm_adapter:
        type: azure_openai
        params:
          resource_name: "my-openai-eastus"
          deployment_id: "gpt-4"
      tools:
        - name: search
          description: "Search the web"
```

### Health Checks

Monitor Azure OpenAI connectivity:

```python
status = await adapter.ahealth_check()
print(f"Status: {status.status}")
print(f"Latency: {status.latency_ms}ms")
print(f"Details: {status.details}")
```

### Fine-Tuned Models

Use your fine-tuned Azure OpenAI deployments:

```yaml
adapter:
  type: azure_openai
  params:
    resource_name: "my-openai-eastus"
    deployment_id: "my-finetuned-gpt-35-turbo"  # Your custom deployment
    api_version: "2024-02-15-preview"
```

## Examples

### Customer Support Agent

```yaml
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: azure-support-agent
spec:
  nodes:
    - kind: agent_node
      metadata:
        name: support_agent
      spec:
        llm_adapter:
          type: azure_openai
          params:
            resource_name: "support-openai"
            deployment_id: "gpt-4"
            temperature: 0.5
        initial_prompt_template: |
          You are a customer support agent.
          Customer query: {{customer_query}}
        max_steps: 10
        tools:
          - name: lookup_order
          - name: process_refund
      dependencies: []
```

### Multi-Step Analysis with Embeddings

```yaml
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: azure-analysis-workflow
spec:
  nodes:
    # Text generation node
    - kind: llm_node
      metadata:
        name: extract_entities
      spec:
        adapter:
          type: azure_openai
          params:
            resource_name: "analytics-openai"
            deployment_id: "gpt-35-turbo"
        prompt_template: "Extract entities from: {{input}}"
      dependencies: []

    # Embedding generation (using same adapter)
    - kind: function_node
      metadata:
        name: generate_embeddings
      spec:
        fn: "myapp.utils.embed_text"  # Uses same Azure adapter
        adapter:
          type: azure_openai
          params:
            resource_name: "analytics-openai"
            embedding_deployment_id: "text-embedding-3-small"
            embedding_dimensions: 1536
      dependencies: [extract_entities]

    - kind: llm_node
      metadata:
        name: summarize
      spec:
        adapter:
          type: azure_openai
          params:
            resource_name: "analytics-openai"
            deployment_id: "gpt-4"
        prompt_template: "Summarize entities: {{extract_entities.result}}"
      dependencies: [extract_entities]
```

## Error Handling

The adapter handles common Azure OpenAI errors gracefully:

- **Authentication failures** - Returns `None` with logged error
- **Rate limiting** - Built-in timeout and retry support
- **Network issues** - Configurable timeout (default 30s)
- **Invalid deployments** - Clear error messages

## Testing

Run tests with pytest:

```bash
cd hexdag_plugins/azure
uv run pytest tests/ -v
```

## Azure OpenAI Setup

### 1. Create Azure OpenAI Resource

```bash
# Using Azure CLI
az cognitiveservices account create \
  --name my-openai-eastus \
  --resource-group my-resource-group \
  --kind OpenAI \
  --sku S0 \
  --location eastus
```

### 2. Deploy a Model

```bash
az cognitiveservices account deployment create \
  --name my-openai-eastus \
  --resource-group my-resource-group \
  --deployment-name gpt-4 \
  --model-name gpt-4 \
  --model-version "0613" \
  --model-format OpenAI \
  --sku-name "Standard" \
  --sku-capacity 1
```

### 3. Get API Key

```bash
az cognitiveservices account keys list \
  --name my-openai-eastus \
  --resource-group my-resource-group
```

## Embedding Support

Generate embeddings using Azure OpenAI's text-embedding models. The adapter supports both unified (LLM + embeddings) and pure embedding use cases.

### Unified LLM + Embeddings

```python
from hexdag_plugins.azure import AzureOpenAIAdapter

# Single adapter for both capabilities
adapter = AzureOpenAIAdapter(
    resource_name="my-openai-eastus",
    deployment_id="gpt-4",  # For text generation
    embedding_deployment_id="text-embedding-3-small",  # For embeddings
    embedding_dimensions=1536,  # Optional: reduce dimensions
)

# Text generation
response = await adapter.aresponse(messages)

# Embeddings
embedding = await adapter.aembed("Hello, world!")
embeddings = await adapter.aembed_batch(["Doc 1", "Doc 2", "Doc 3"])
```

### Pure Embedding Adapter

For embedding-only use cases:

```python
# deployment_id still required by LLM protocol, but won't be used
adapter = AzureOpenAIAdapter(
    resource_name="my-openai-eastus",
    deployment_id="gpt-4",  # Required but not used
    embedding_deployment_id="text-embedding-3-small",
)

# Only use embedding methods
embedding = await adapter.aembed("Document text")
embeddings = await adapter.aembed_batch(documents)
```

### YAML Configuration

```yaml
adapter:
  type: azure_openai
  params:
    resource_name: "my-openai-eastus"
    embedding_deployment_id: "text-embedding-3-small"
    embedding_dimensions: 1536
```

### Supported Embedding Models

| Model | Dimensions | Description |
|-------|-----------|-------------|
| `text-embedding-3-small` | 1536 (default) | Cost-effective, high performance |
| `text-embedding-3-large` | 3072 (default) | Highest quality embeddings |
| `text-embedding-ada-002` | 1536 | Legacy model, still supported |

**Note:** Image embeddings are not currently supported by Azure OpenAI's embeddings API. For multimodal use cases, consider using vision models with `aresponse_with_vision()`.

## Supported Models

### Text Generation Models

- **GPT-4** - Most capable model (all versions)
- **GPT-3.5-turbo** - Fast and cost-effective (all versions)
- **Fine-tuned models** - Your custom trained models
- **Future models** - Compatible with new Azure OpenAI releases

### Embedding Models

- **text-embedding-3-small** - 1536 dimensions, cost-effective
- **text-embedding-3-large** - 3072 dimensions, highest quality
- **text-embedding-ada-002** - 1536 dimensions, legacy support

## API Version Support

| API Version | Status | Features |
|-------------|--------|----------|
| `2024-02-15-preview` | ✅ Recommended | Latest features + tool calling |
| `2023-12-01-preview` | ✅ Supported | Stable with tool calling |
| `2023-05-15` | ✅ Supported | Stable without tool calling |

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - See LICENSE file for details

## Support

- **Documentation**: https://hexdag.ai/docs/plugins/azure
- **Issues**: https://github.com/hexdag/hexdag-azure/issues
- **Discussions**: https://github.com/hexdag/hexdag/discussions

## Related

- [hexdag](https://github.com/hexdag/hexdag) - Core framework
- [Azure OpenAI Documentation](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
