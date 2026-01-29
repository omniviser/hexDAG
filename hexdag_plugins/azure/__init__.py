"""Azure plugin for hexDAG framework.

Provides adapters for Azure services:
- AzureOpenAIAdapter: Azure OpenAI for LLM operations
- AzureKeyVaultAdapter: Azure Key Vault for secret management
- AzureCosmosAdapter: Azure Cosmos DB for memory/state
- AzureBlobAdapter: Azure Blob Storage for files
"""

from hexdag_plugins.azure.azure_blob_adapter import AzureBlobAdapter
from hexdag_plugins.azure.azure_cosmos_adapter import AzureCosmosAdapter
from hexdag_plugins.azure.azure_keyvault_adapter import AzureKeyVaultAdapter
from hexdag_plugins.azure.azure_openai_adapter import AzureOpenAIAdapter

__all__ = [
    "AzureOpenAIAdapter",
    "AzureKeyVaultAdapter",
    "AzureCosmosAdapter",
    "AzureBlobAdapter",
]
__version__ = "0.2.0"
