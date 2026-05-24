"""Azure adapters for hexDAG framework.

Provides adapters for Azure cloud services:

- ``AzureOpenAIAdapter``: Azure OpenAI for LLM operations
- ``AzureKeyVaultAdapter``: Azure Key Vault for secret management
- ``AzureCosmosAdapter``: Azure Cosmos DB for memory/state
- ``AzureBlobAdapter``: Azure Blob Storage for files
"""

from hexdag_plugins.azure.adapters.blob import AzureBlobAdapter
from hexdag_plugins.azure.adapters.cosmos import AzureCosmosAdapter
from hexdag_plugins.azure.adapters.keyvault import AzureKeyVaultAdapter
from hexdag_plugins.azure.adapters.openai import AzureOpenAIAdapter

__all__ = [
    "AzureOpenAIAdapter",
    "AzureKeyVaultAdapter",
    "AzureCosmosAdapter",
    "AzureBlobAdapter",
]
