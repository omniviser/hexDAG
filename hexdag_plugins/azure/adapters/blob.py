"""Azure Blob Storage adapter for hexDAG framework.

Provides file storage and retrieval for pipelines and agents.
"""

import os
import time
from typing import Any

from hexdag.core.ports.file_storage import FileStoragePort
from hexdag.core.ports.healthcheck import HealthStatus


class AzureBlobAdapter(FileStoragePort):
    """Azure Blob Storage adapter for file operations.

    Provides scalable file storage for documents, artifacts, and pipeline
    outputs using Azure Blob Storage.

    Parameters
    ----------
    connection_string : str
        Azure Storage connection string (auto-resolved from AZURE_STORAGE_CONNECTION_STRING)
    container_name : str
        Blob container name (default: "hexdag")
    account_url : str, optional
        Storage account URL (for managed identity auth)
    use_managed_identity : bool
        Use Managed Identity instead of connection string (default: False)

    Examples
    --------
    YAML configuration::

        spec:
          ports:
            storage:
              adapter: hexdag_plugins.azure.AzureBlobAdapter
              config:
                container_name: "pipeline-artifacts"

    With managed identity::

        spec:
          ports:
            storage:
              adapter: hexdag_plugins.azure.AzureBlobAdapter
              config:
                account_url: "https://mystorageaccount.blob.core.windows.net"
                container_name: "pipeline-artifacts"
                use_managed_identity: true

    Python usage::

        from hexdag_plugins.azure import AzureBlobAdapter

        adapter = AzureBlobAdapter(
            connection_string="...",  # or auto-resolved
            container_name="pipeline-artifacts"
        )

        # Upload file
        await adapter.aupload("reports/output.json", json_bytes)

        # Download file
        content = await adapter.adownload("reports/output.json")

        # List files
        files = await adapter.alist("reports/")
    """

    _hexdag_icon = "HardDrive"
    _hexdag_color = "#0078d4"  # Azure blue

    def __init__(
        self,
        connection_string: str | None = None,
        container_name: str = "hexdag",
        account_url: str | None = None,
        use_managed_identity: bool = False,
    ):
        """Initialize Azure Blob Storage adapter.

        Args
        ----
            connection_string: Azure Storage connection string
                (auto-resolved from AZURE_STORAGE_CONNECTION_STRING)
            container_name: Blob container name (default: "hexdag")
            account_url: Storage account URL (for managed identity)
            use_managed_identity: Use Managed Identity (default: False)
        """
        self.connection_string = connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.container_name = container_name
        self.account_url = account_url
        self.use_managed_identity = use_managed_identity

        self._container_client = None

    async def _get_container(self):
        """Get or create blob container client."""
        if self._container_client is None:
            try:
                from azure.storage.blob.aio import BlobServiceClient
            except ImportError as e:
                raise ImportError(
                    "Azure Storage SDK not installed. Install with: pip install azure-storage-blob"
                ) from e

            if self.use_managed_identity:
                if not self.account_url:
                    raise ValueError("account_url is required when use_managed_identity=True")
                try:
                    from azure.identity.aio import DefaultAzureCredential

                    credential = DefaultAzureCredential()
                    service_client = BlobServiceClient(
                        account_url=self.account_url, credential=credential
                    )
                except ImportError as e:
                    raise ImportError(
                        "Azure Identity SDK not installed. Install with: pip install azure-identity"
                    ) from e
            else:
                if not self.connection_string:
                    raise ValueError(
                        "connection_string is required when use_managed_identity=False"
                    )
                service_client = BlobServiceClient.from_connection_string(self.connection_string)

            # Create container if not exists
            self._container_client = service_client.get_container_client(self.container_name)
            try:
                await self._container_client.create_container()
            except Exception:  # noqa: SIM105 - contextlib.suppress doesn't work with async
                # Container might already exist
                pass

        return self._container_client

    async def aupload(
        self,
        blob_name: str,
        data: bytes | str,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
        overwrite: bool = True,
    ) -> str:
        """Upload data to blob storage.

        Args
        ----
            blob_name: Name/path of the blob
            data: Data to upload (bytes or string)
            content_type: MIME type of the content
            metadata: Optional blob metadata
            overwrite: Overwrite existing blob (default: True)

        Returns
        -------
            URL of the uploaded blob
        """
        container = await self._get_container()

        if isinstance(data, str):
            data = data.encode("utf-8")
            if content_type is None:
                content_type = "text/plain"

        blob_client = container.get_blob_client(blob_name)

        await blob_client.upload_blob(
            data,
            content_type=content_type,
            metadata=metadata,
            overwrite=overwrite,
        )

        return blob_client.url

    async def adownload(self, blob_name: str) -> bytes:
        """Download blob content.

        Args
        ----
            blob_name: Name/path of the blob

        Returns
        -------
            Blob content as bytes

        Raises
        ------
            FileNotFoundError: If blob doesn't exist
        """
        container = await self._get_container()
        blob_client = container.get_blob_client(blob_name)

        try:
            stream = await blob_client.download_blob()
            return await stream.readall()
        except Exception as e:
            if "BlobNotFound" in str(e):
                raise FileNotFoundError(f"Blob '{blob_name}' not found") from e
            raise

    async def adownload_text(self, blob_name: str, encoding: str = "utf-8") -> str:
        """Download blob content as text.

        Args
        ----
            blob_name: Name/path of the blob
            encoding: Text encoding (default: "utf-8")

        Returns
        -------
            Blob content as string
        """
        content = await self.adownload(blob_name)
        return content.decode(encoding)

    async def adelete(self, blob_name: str) -> bool:
        """Delete a blob.

        Args
        ----
            blob_name: Name/path of the blob

        Returns
        -------
            True if deleted, False if not found
        """
        container = await self._get_container()
        blob_client = container.get_blob_client(blob_name)

        try:
            await blob_client.delete_blob()
            return True
        except Exception:
            return False

    async def aexists(self, blob_name: str) -> bool:
        """Check if blob exists.

        Args
        ----
            blob_name: Name/path of the blob

        Returns
        -------
            True if exists, False otherwise
        """
        container = await self._get_container()
        blob_client = container.get_blob_client(blob_name)
        return await blob_client.exists()

    async def alist(
        self,
        prefix: str | None = None,
        max_results: int | None = None,
    ) -> list[dict[str, Any]]:
        """List blobs in container.

        Args
        ----
            prefix: Optional prefix to filter blobs
            max_results: Maximum number of results

        Returns
        -------
            List of blob info dicts with name, size, last_modified, etc.
        """
        container = await self._get_container()

        results = []
        count = 0

        async for blob in container.list_blobs(name_starts_with=prefix):
            results.append(
                {
                    "name": blob.name,
                    "size": blob.size,
                    "last_modified": blob.last_modified,
                    "content_type": blob.content_settings.content_type
                    if blob.content_settings
                    else None,
                    "metadata": blob.metadata,
                }
            )
            count += 1
            if max_results and count >= max_results:
                break

        return results

    async def acopy(self, source_blob: str, dest_blob: str) -> str:
        """Copy blob within container.

        Args
        ----
            source_blob: Source blob name
            dest_blob: Destination blob name

        Returns
        -------
            URL of the copied blob
        """
        container = await self._get_container()
        source_client = container.get_blob_client(source_blob)
        dest_client = container.get_blob_client(dest_blob)

        await dest_client.start_copy_from_url(source_client.url)
        return dest_client.url

    async def aget_url(self, blob_name: str, expiry_hours: int = 1) -> str:
        """Get a SAS URL for blob access.

        Args
        ----
            blob_name: Name/path of the blob
            expiry_hours: URL expiry time in hours

        Returns
        -------
            SAS URL for blob access
        """
        from datetime import datetime, timedelta

        try:
            from azure.storage.blob import BlobSasPermissions, generate_blob_sas
        except ImportError as e:
            raise ImportError(
                "Azure Storage SDK not installed. Install with: pip install azure-storage-blob"
            ) from e

        container = await self._get_container()
        blob_client = container.get_blob_client(blob_name)

        # Generate SAS token
        sas_token = generate_blob_sas(
            account_name=blob_client.account_name,
            container_name=self.container_name,
            blob_name=blob_name,
            account_key=self._get_account_key(),
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=expiry_hours),
        )

        return f"{blob_client.url}?{sas_token}"

    def _get_account_key(self) -> str | None:
        """Extract account key from connection string."""
        if not self.connection_string:
            return None

        for part in self.connection_string.split(";"):
            if part.startswith("AccountKey="):
                return part[11:]
        return None

    async def aupload_json(
        self,
        blob_name: str,
        data: dict | list,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Upload JSON data to blob storage.

        Args
        ----
            blob_name: Name/path of the blob
            data: JSON-serializable data
            metadata: Optional blob metadata

        Returns
        -------
            URL of the uploaded blob
        """
        import json as json_module

        json_str = json_module.dumps(data, indent=2, default=str)
        return await self.aupload(
            blob_name,
            json_str.encode("utf-8"),
            content_type="application/json",
            metadata=metadata,
        )

    async def adownload_json(self, blob_name: str) -> dict | list:
        """Download and parse JSON blob.

        Args
        ----
            blob_name: Name/path of the blob

        Returns
        -------
            Parsed JSON data
        """
        import json as json_module

        content = await self.adownload_text(blob_name)
        return json_module.loads(content)

    async def ahealth_check(self) -> HealthStatus:
        """Check Azure Blob Storage connectivity.

        Returns
        -------
            HealthStatus with connectivity details
        """
        try:
            start_time = time.time()
            container = await self._get_container()

            # Count blobs to verify access
            count = 0
            async for _ in container.list_blobs():
                count += 1
                if count >= 10:  # Sample only
                    break

            latency_ms = (time.time() - start_time) * 1000

            return HealthStatus(
                status="healthy",
                adapter_name="AzureBlob",
                latency_ms=latency_ms,
                details={
                    "container": self.container_name,
                    "sample_blob_count": count,
                },
            )

        except Exception as e:
            return HealthStatus(
                status="unhealthy",
                adapter_name="AzureBlob",
                latency_ms=0.0,
                details={"error": str(e)},
            )

    async def aclose(self) -> None:
        """Close the blob storage client."""
        if self._container_client:
            await self._container_client.close()
            self._container_client = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize adapter configuration (excluding secrets)."""
        return {
            "container_name": self.container_name,
            "account_url": self.account_url,
            "use_managed_identity": self.use_managed_identity,
        }
