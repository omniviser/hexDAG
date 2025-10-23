"""Tests for local file storage adapter."""

import pytest

from hexdag_plugins.storage.file import LocalFileStorage


class TestLocalFileStorage:
    """Test suite for LocalFileStorage adapter."""

    @pytest.fixture
    async def storage(self, tmp_path):
        """Create a test storage instance."""
        storage = LocalFileStorage(base_path=str(tmp_path))
        yield storage

    @pytest.fixture
    def test_file(self, tmp_path):
        """Create a test file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")
        return test_file

    @pytest.mark.asyncio
    async def test_initialization(self, tmp_path):
        """Test storage initialization."""
        storage = LocalFileStorage(base_path=str(tmp_path))
        assert storage._base_path == tmp_path
        assert tmp_path.exists()

    @pytest.mark.asyncio
    async def test_upload(self, storage, test_file, tmp_path):
        """Test file upload."""
        result = await storage.aupload(str(test_file), "docs/test.txt")

        assert result["uploaded"] is True
        assert result["remote_path"] == "docs/test.txt"
        assert (tmp_path / "docs/test.txt").exists()

    @pytest.mark.asyncio
    async def test_download(self, storage, test_file, tmp_path):
        """Test file download."""
        # First upload a file
        await storage.aupload(str(test_file), "docs/test.txt")

        # Download to a different location
        download_path = tmp_path / "downloaded.txt"
        result = await storage.adownload("docs/test.txt", str(download_path))

        assert result["downloaded"] is True
        assert download_path.exists()
        assert download_path.read_text() == "Hello, World!"

    @pytest.mark.asyncio
    async def test_exists(self, storage, test_file):
        """Test file existence check."""
        # File doesn't exist yet
        assert await storage.aexists("docs/test.txt") is False

        # Upload file
        await storage.aupload(str(test_file), "docs/test.txt")

        # Now it exists
        assert await storage.aexists("docs/test.txt") is True

    @pytest.mark.asyncio
    async def test_delete(self, storage, test_file):
        """Test file deletion."""
        # Upload file
        await storage.aupload(str(test_file), "docs/test.txt")
        assert await storage.aexists("docs/test.txt") is True

        # Delete file
        result = await storage.adelete("docs/test.txt")

        assert result["deleted"] is True
        assert await storage.aexists("docs/test.txt") is False

    @pytest.mark.asyncio
    async def test_list(self, storage, test_file):
        """Test file listing."""
        # Upload multiple files
        await storage.aupload(str(test_file), "docs/file1.txt")
        await storage.aupload(str(test_file), "docs/file2.txt")
        await storage.aupload(str(test_file), "images/pic.jpg")

        # List all files
        all_files = await storage.alist()
        assert len(all_files) == 3
        assert "docs/file1.txt" in all_files
        assert "docs/file2.txt" in all_files
        assert "images/pic.jpg" in all_files

        # List with prefix
        docs_files = await storage.alist(prefix="docs/")
        assert len(docs_files) == 2
        assert all(f.startswith("docs/") for f in docs_files)

    @pytest.mark.asyncio
    async def test_get_metadata(self, storage, test_file):
        """Test getting file metadata."""
        await storage.aupload(str(test_file), "test.txt")

        metadata = await storage.aget_metadata("test.txt")

        assert metadata["path"] == "test.txt"
        assert metadata["size_bytes"] > 0
        assert metadata["is_file"] is True
        assert "modified_time" in metadata
        assert "created_time" in metadata

    @pytest.mark.asyncio
    async def test_health_check(self, storage):
        """Test health check."""
        health = await storage.ahealth_check()

        assert health.status == "healthy"
        assert health.adapter_name == "local_file_storage"
        assert health.port_name == "file_storage"
        assert health.details["writable"] is True

    @pytest.mark.asyncio
    async def test_upload_creates_directories(self, storage, test_file):
        """Test that upload creates nested directories."""
        result = await storage.aupload(str(test_file), "a/b/c/test.txt")

        assert result["uploaded"] is True
        assert await storage.aexists("a/b/c/test.txt") is True

    @pytest.mark.asyncio
    async def test_upload_nonexistent_file(self, storage):
        """Test uploading a nonexistent file."""
        with pytest.raises(FileNotFoundError):
            await storage.aupload("/nonexistent/file.txt", "test.txt")

    @pytest.mark.asyncio
    async def test_download_nonexistent_file(self, storage, tmp_path):
        """Test downloading a nonexistent file."""
        with pytest.raises(FileNotFoundError):
            await storage.adownload("nonexistent.txt", str(tmp_path / "out.txt"))

    @pytest.mark.asyncio
    async def test_delete_nonexistent_file(self, storage):
        """Test deleting a nonexistent file."""
        with pytest.raises(FileNotFoundError):
            await storage.adelete("nonexistent.txt")

    @pytest.mark.asyncio
    async def test_get_metadata_nonexistent_file(self, storage):
        """Test getting metadata for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            await storage.aget_metadata("nonexistent.txt")

    @pytest.mark.asyncio
    async def test_repr(self, tmp_path):
        """Test string representation."""
        storage = LocalFileStorage(base_path=str(tmp_path))
        repr_str = repr(storage)

        assert "LocalFileStorage" in repr_str
        assert str(tmp_path) in repr_str
