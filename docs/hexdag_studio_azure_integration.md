# hexdag Studio - Azure Integration Stack

This document outlines the Azure integration architecture for deploying hexdag studio pipelines to production.

## Existing Azure Plugin

hexdag already has a mature Azure plugin at `hexdag_plugins/azure/`:

- **`AzureOpenAIAdapter`** - Full-featured Azure OpenAI adapter with:
  - Text generation (`aresponse`)
  - Tool calling (`aresponse_with_tools`)
  - Embeddings (`aembed`, `aembed_batch`)
  - Health checks (`ahealth_check`)
  - Automatic secret resolution from `AZURE_OPENAI_API_KEY`

```yaml
# Using existing Azure OpenAI adapter
spec:
  ports:
    llm:
      adapter: azure_openai
      config:
        resource_name: "my-openai-resource"
        deployment_id: "gpt-4"
        api_version: "2024-02-15-preview"
```

This document covers **additional Azure services** and **deployment strategies** for hexdag studio.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        hexdag Studio UI                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │ Visual      │  │   YAML      │  │  Export     │  │  Deploy     │ │
│  │ Canvas      │  │   Editor    │  │  Project    │  │  to Azure   │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Azure Deployment Stack                           │
│                                                                      │
│  ┌──────────────────┐       ┌──────────────────┐                   │
│  │ Azure Container  │       │ Azure Functions  │                   │
│  │ Apps (ACA)       │       │ (Serverless)     │                   │
│  │ - Pipeline API   │       │ - Event triggers │                   │
│  │ - Studio Server  │       │ - Scheduled runs │                   │
│  └──────────────────┘       └──────────────────┘                   │
│           │                          │                              │
│           └──────────┬───────────────┘                              │
│                      ▼                                              │
│  ┌──────────────────────────────────────────────────────┐         │
│  │              Azure Service Integrations               │         │
│  │                                                       │         │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │         │
│  │  │ Key Vault   │  │ Cosmos DB   │  │ Blob        │  │         │
│  │  │ (Secrets)   │  │ (State)     │  │ Storage     │  │         │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  │         │
│  │                                                       │         │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │         │
│  │  │ Azure       │  │ Service Bus │  │ App         │  │         │
│  │  │ OpenAI      │  │ (Queues)    │  │ Insights    │  │         │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  │         │
│  └──────────────────────────────────────────────────────┘         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Azure Key Vault (Secrets Management) - IMPLEMENTED

> **Note:** This is already implemented in `hexdag_plugins/azure/azure_keyvault_adapter.py`

hexdag's deferred secret resolution integrates with Azure Key Vault:

```yaml
spec:
  ports:
    secret:
      adapter: azure_keyvault
      config:
        vault_url: "https://my-vault.vault.azure.net"
        use_managed_identity: true  # Recommended for Azure deployments
```

**Installation:**

```bash
pip install hexdag-azure[keyvault]
```

**Python usage:**

```python
from hexdag_plugins.azure import AzureKeyVaultAdapter

# With managed identity (recommended for Azure deployments)
adapter = AzureKeyVaultAdapter(
    vault_url="https://my-vault.vault.azure.net",
    use_managed_identity=True
)

# With service principal (for local dev)
adapter = AzureKeyVaultAdapter(
    vault_url="https://my-vault.vault.azure.net",
    use_managed_identity=False,
    tenant_id="...",
    client_id="...",
    client_secret="..."
)

# Retrieve secrets
api_key = await adapter.aget("OPENAI-API-KEY")
db_password = await adapter.aget("DB-PASSWORD")

# Batch retrieval
secrets = await adapter.aget_batch(["SECRET1", "SECRET2"])

# Set/delete secrets
await adapter.aset("NEW-SECRET", "value")
await adapter.adelete("OLD-SECRET")

# List all secrets
names = await adapter.alist()
```

**Features:**
- In-memory caching with configurable TTL (default 300s)
- Supports both Managed Identity and Service Principal auth
- Health checks for connectivity monitoring

### 2. Azure OpenAI (LLM Port) - ALREADY IMPLEMENTED

> **Note:** This is already implemented in `hexdag_plugins/azure/azure_openai_adapter.py`

```yaml
# pipeline.yaml - Using existing AzureOpenAIAdapter
spec:
  ports:
    llm:
      adapter: azure_openai
      config:
        resource_name: "my-openai-resource"
        deployment_id: "gpt-4"
        api_version: "2024-02-15-preview"
        temperature: 0.7
        # Optional: for embeddings
        embedding_deployment_id: "text-embedding-3-small"
```

**Python usage:**

```python
from hexdag_plugins.azure import AzureOpenAIAdapter

# Text generation + embeddings
adapter = AzureOpenAIAdapter(
    api_key="your-key",  # or auto-resolved from AZURE_OPENAI_API_KEY
    resource_name="my-openai-resource",
    deployment_id="gpt-4",
    embedding_deployment_id="text-embedding-3-small",
)

# Text generation
messages = [{"role": "user", "content": "Hello"}]
response = await adapter.aresponse(messages)

# Tool calling
response = await adapter.aresponse_with_tools(messages, tools=[...])

# Embeddings
embedding = await adapter.aembed("Hello, world!")
embeddings = await adapter.aembed_batch(["Text 1", "Text 2"])
```

### 3. Azure Cosmos DB (State & Memory) - IMPLEMENTED

> **Note:** This is already implemented in `hexdag_plugins/azure/azure_cosmos_adapter.py`

For persistent agent memory and pipeline state:

```yaml
spec:
  ports:
    memory:
      adapter: azure_cosmos
      config:
        endpoint: ${AZURE_COSMOS_ENDPOINT}
        database_name: hexdag
        container_name: agent_memory
```

**Installation:**

```bash
pip install hexdag-azure[cosmos]
```

**Python usage:**

```python
from hexdag_plugins.azure import AzureCosmosAdapter

adapter = AzureCosmosAdapter(
    endpoint="https://my-cosmos.documents.azure.com:443/",
    key="...",  # or auto-resolved from AZURE_COSMOS_KEY
    database_name="hexdag",
    container_name="agent_memory"
)

# Store agent memory
await adapter.astore("agent-123", {"context": "...", "history": [...]})

# Retrieve memory
memory = await adapter.aretrieve("agent-123")

# Store conversation history
await adapter.astore_conversation("agent-123", messages, session_id="session-1")

# Search memories
results = await adapter.asearch("user query", top_k=5)
```

### 4. Azure Blob Storage (File Operations) - IMPLEMENTED

> **Note:** This is already implemented in `hexdag_plugins/azure/azure_blob_adapter.py`

For document processing and artifact storage:

```yaml
spec:
  ports:
    storage:
      adapter: azure_blob
      config:
        container_name: pipeline-artifacts
        # Connection string auto-resolved from AZURE_STORAGE_CONNECTION_STRING
        # OR use managed identity:
        account_url: "https://mystorageaccount.blob.core.windows.net"
        use_managed_identity: true
```

**Installation:**

```bash
pip install hexdag-azure[blob]
```

**Python usage:**

```python
from hexdag_plugins.azure import AzureBlobAdapter

# With connection string
adapter = AzureBlobAdapter(
    connection_string="...",  # or auto-resolved from env
    container_name="pipeline-artifacts"
)

# With managed identity
adapter = AzureBlobAdapter(
    account_url="https://mystorageaccount.blob.core.windows.net",
    container_name="pipeline-artifacts",
    use_managed_identity=True
)

# Upload files
url = await adapter.aupload("reports/output.json", json_bytes)

# Download files
content = await adapter.adownload("reports/output.json")
text = await adapter.adownload_text("reports/output.txt")

# JSON convenience methods
await adapter.aupload_json("data.json", {"key": "value"})
data = await adapter.adownload_json("data.json")

# List files
files = await adapter.alist(prefix="reports/")

# Generate SAS URLs
sas_url = await adapter.aget_url("file.pdf", expiry_hours=1)
```

### 5. Azure Service Bus (Event-Driven Pipelines)

For triggering pipelines from events:

```yaml
# azure-trigger.yaml
apiVersion: hexdag/v1
kind: Trigger
metadata:
  name: order-processor
spec:
  type: service_bus
  config:
    connection_string: ${SERVICE_BUS_CONNECTION_STRING}
    queue_name: orders
    max_concurrent_calls: 5
  pipeline: order-processing-pipeline.yaml
```

### 6. Azure Container Apps (Deployment)

Deploy hexdag pipelines as containerized services:

```yaml
# azure-container-app.yaml
name: hexdag-pipeline
properties:
  configuration:
    activeRevisionsMode: Single
    ingress:
      external: true
      targetPort: 8000
    secrets:
      - name: openai-key
        keyVaultUrl: https://my-vault.vault.azure.net/secrets/OPENAI-API-KEY
  template:
    containers:
      - name: hexdag-pipeline
        image: myregistry.azurecr.io/hexdag-pipeline:latest
        resources:
          cpu: 1
          memory: 2Gi
        env:
          - name: OPENAI_API_KEY
            secretRef: openai-key
```

### 7. Azure App Insights (Observability)

Comprehensive logging and monitoring:

```python
# hexdag/builtin/observers/azure_appinsights.py
from opencensus.ext.azure.log_exporter import AzureLogHandler

class AppInsightsObserver:
    """Azure Application Insights observer for pipeline telemetry."""

    def __init__(self, connection_string: str):
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(
            AzureLogHandler(connection_string=connection_string)
        )

    async def on_node_started(self, event: NodeStarted):
        self.logger.info(
            "Node started",
            extra={
                "custom_dimensions": {
                    "node_id": event.node_id,
                    "pipeline_id": event.pipeline_id
                }
            }
        )

    async def on_node_completed(self, event: NodeCompleted):
        self.logger.info(
            "Node completed",
            extra={
                "custom_dimensions": {
                    "node_id": event.node_id,
                    "duration_ms": event.duration_ms,
                    "pipeline_id": event.pipeline_id
                }
            }
        )
```

## Deployment Options

### Option 1: Azure Container Apps (Recommended)

Best for production workloads with auto-scaling:

```bash
# Deploy from hexdag studio export
hexdag deploy azure \
  --pipeline my-pipeline.yaml \
  --target container-apps \
  --resource-group hexdag-rg \
  --environment hexdag-env
```

### Option 2: Azure Functions (Serverless)

Best for event-driven, infrequent executions:

```bash
hexdag deploy azure \
  --pipeline my-pipeline.yaml \
  --target functions \
  --resource-group hexdag-rg \
  --function-app hexdag-funcs
```

### Option 3: Azure Kubernetes Service (Enterprise)

Best for complex orchestration with existing K8s infrastructure:

```bash
hexdag deploy azure \
  --pipeline my-pipeline.yaml \
  --target aks \
  --cluster hexdag-cluster \
  --namespace production
```

## hexdag Studio UI Integration

### Deploy Button

Add a "Deploy to Azure" button in hexdag studio:

```typescript
// hexdag/studio/ui/src/components/AzureDeployModal.tsx
export function AzureDeployModal({ pipeline, onClose }) {
  const [target, setTarget] = useState<'container-apps' | 'functions' | 'aks'>('container-apps')
  const [resourceGroup, setResourceGroup] = useState('')

  const handleDeploy = async () => {
    const response = await fetch('/api/deploy/azure', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        pipeline,
        target,
        resource_group: resourceGroup,
        // ... other config
      })
    })
    // Handle response
  }

  return (
    <Modal>
      <h2>Deploy to Azure</h2>
      <select value={target} onChange={e => setTarget(e.target.value)}>
        <option value="container-apps">Azure Container Apps</option>
        <option value="functions">Azure Functions</option>
        <option value="aks">Azure Kubernetes Service</option>
      </select>
      {/* Resource group, environment, etc. */}
      <button onClick={handleDeploy}>Deploy</button>
    </Modal>
  )
}
```

### Backend API

```python
# hexdag/studio/server/routes/deploy.py
from fastapi import APIRouter
from azure.identity import DefaultAzureCredential
from azure.mgmt.containerinstance import ContainerInstanceManagementClient

router = APIRouter(prefix="/deploy", tags=["deploy"])

@router.post("/azure")
async def deploy_to_azure(request: AzureDeployRequest):
    """Deploy pipeline to Azure."""

    # 1. Export pipeline as Docker image
    export_result = await export_pipeline(request.pipeline)

    # 2. Build and push to ACR
    await build_and_push_image(export_result, request.acr_name)

    # 3. Deploy to target
    if request.target == "container-apps":
        return await deploy_to_container_apps(request)
    elif request.target == "functions":
        return await deploy_to_functions(request)
    elif request.target == "aks":
        return await deploy_to_aks(request)
```

## Environment Configuration

### Development
```yaml
# environments/dev.yaml
spec:
  ports:
    llm:
      adapter: mock_llm  # Use mocks for testing
    memory:
      adapter: in_memory
    storage:
      adapter: local_filesystem
```

### Staging
```yaml
# environments/staging.yaml
spec:
  ports:
    llm:
      adapter: azure_openai
      config:
        endpoint: ${AZURE_OPENAI_STAGING_ENDPOINT}
        deployment_name: gpt-35-turbo  # Cheaper model
    memory:
      adapter: cosmos_memory
      config:
        database_name: hexdag-staging
```

### Production
```yaml
# environments/prod.yaml
spec:
  ports:
    llm:
      adapter: azure_openai
      config:
        endpoint: ${AZURE_OPENAI_PROD_ENDPOINT}
        deployment_name: gpt-4-turbo
    memory:
      adapter: cosmos_memory
      config:
        database_name: hexdag-prod
    secret:
      adapter: azure_keyvault
      config:
        vault_url: https://hexdag-prod.vault.azure.net
```

## Security Best Practices

1. **Use Managed Identity** - Avoid storing secrets in code
2. **Key Vault Integration** - All secrets via Azure Key Vault
3. **Network Isolation** - VNet integration for Container Apps
4. **RBAC** - Fine-grained access control
5. **Private Endpoints** - No public exposure for data stores

## Implementation Status

| Adapter | Status | Installation |
|---------|--------|--------------|
| `AzureOpenAIAdapter` | ✅ Complete | `pip install hexdag-azure` |
| `AzureKeyVaultAdapter` | ✅ Complete | `pip install hexdag-azure[keyvault]` |
| `AzureCosmosAdapter` | ✅ Complete | `pip install hexdag-azure[cosmos]` |
| `AzureBlobAdapter` | ✅ Complete | `pip install hexdag-azure[blob]` |
| Service Bus Adapter | ❌ Not implemented | - |
| App Insights Observer | ❌ Not implemented | - |

**Install all Azure adapters:**

```bash
pip install hexdag-azure[all]
```

## Next Steps

1. ~~Implement `AzureKeyVaultAdapter`~~ ✅ Done
2. ~~Implement `AzureOpenAIAdapter`~~ ✅ Done
3. ~~Implement `CosmosMemoryAdapter`~~ ✅ Done
4. ~~Implement `AzureBlobAdapter`~~ ✅ Done
5. Add "Deploy to Azure" UI in hexdag studio
6. Create Bicep/ARM templates for infrastructure
7. Add Azure Pipelines CI/CD templates
8. Implement Service Bus adapter for event-driven pipelines
9. Implement App Insights observer for telemetry
