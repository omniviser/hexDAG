"""Azure plugin for hexDAG framework.

Plugin Structure
----------------
::

    azure/
    ├── __init__.py          # This file - re-exports from adapters/
    ├── adapters/            # Adapter implementations
    │   ├── __init__.py
    │   ├── openai.py        # AzureOpenAIAdapter
    │   ├── keyvault.py      # AzureKeyVaultAdapter
    │   ├── cosmos.py        # AzureCosmosAdapter
    │   └── blob.py          # AzureBlobAdapter
    └── tests/               # Test files

Available Adapters
------------------
- ``AzureOpenAIAdapter``: Azure OpenAI for LLM operations (inherits ``LLM``)
- ``AzureKeyVaultAdapter``: Azure Key Vault for secret management (inherits ``SecretPort``)
- ``AzureCosmosAdapter``: Azure Cosmos DB for memory/state (inherits ``Memory``)
- ``AzureBlobAdapter``: Azure Blob Storage for files (inherits ``FileStoragePort``)
"""

from hexdag_plugins.azure.adapters import (
    AzureBlobAdapter,
    AzureCosmosAdapter,
    AzureKeyVaultAdapter,
    AzureOpenAIAdapter,
)

__all__ = [
    "AzureOpenAIAdapter",
    "AzureKeyVaultAdapter",
    "AzureCosmosAdapter",
    "AzureBlobAdapter",
]
__version__ = "0.2.0"
