"""File-based Memory adapter for JSON/YAML/pickle storage.

This adapter provides file-based key-value storage, allowing Memory Port
to work with various file formats.

SECURITY WARNING: The pickle format can execute arbitrary code during
deserialization. Only use pickle with trusted data sources. For untrusted
data, use JSON or TEXT formats instead.
"""

import json
import pickle  # nosec B403 - Pickle usage documented, user must choose format
from enum import StrEnum
from pathlib import Path
from typing import Any

from hexdag.core.logging import get_logger
from hexdag.core.ports.memory import Memory

logger = get_logger(__name__)


class FileFormat(StrEnum):
    """Supported file formats for storage.

    Security Notes
    --------------
    - JSON: Safe for untrusted data, human-readable
    - TEXT: Safe for untrusted data, stores as plain text
    - PICKLE: **UNSAFE for untrusted data** - can execute arbitrary code
      Only use pickle with data you control or trust completely.
    """

    JSON = "json"
    PICKLE = "pickle"
    TEXT = "text"


class FileMemoryAdapter(Memory):
    """Memory adapter backed by file system.

    Provides persistent key-value storage using files, with support for
    multiple formats (JSON, pickle, text). Each key is stored as a separate
    file in the specified directory.

    This adapter is ideal for:
    - Configuration files
    - Pipeline definitions
    - Human-readable checkpoints
    - Data serialization

    Parameters
    ----------
    base_path : str | Path
        Base directory for file storage
    format : FileFormat, default=FileFormat.JSON
        File format to use (json, pickle, text)
    create_dirs : bool, default=True
        Automatically create directory structure
    extension : str | None
        Custom file extension (defaults to format)

    Examples
    --------
    Example usage::

        memory = FileMemoryAdapter(base_path="./data", format="json")
        memory = FileMemoryAdapter(base_path="./cache", format="pickle")
    """

    def __init__(
        self,
        base_path: str | Path = "./memory_store",
        format: FileFormat | str = FileFormat.JSON,
        create_dirs: bool = True,
        extension: str | None = None,
    ) -> None:
        """Initialize file memory adapter.

        Parameters
        ----------
        base_path : str | Path
            Base directory for file storage
        format : FileFormat | str
            File format (json, pickle, text)
        create_dirs : bool
            Automatically create directory if it doesn't exist
        extension : str | None
            Custom file extension (defaults to format name)
        """
        self.base_path = Path(base_path)
        self.format = FileFormat(format) if isinstance(format, str) else format
        self.create_dirs = create_dirs
        self.extension = extension or self.format.value

        if self.create_dirs:
            self.base_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Initialized file storage at '{self.base_path}'")

    def _get_file_path(self, key: str) -> Path:
        """Get file path for a key.

        Parameters
        ----------
        key : str
            Storage key

        Returns
        -------
        Path
            Full file path for the key
        """
        # Sanitize key to be filesystem-safe
        safe_key = key.replace("/", "_").replace(":", "_")
        return self.base_path / f"{safe_key}.{self.extension}"

    def _serialize(self, value: Any) -> bytes | str:
        """Serialize value based on format.

        Parameters
        ----------
        value : Any
            Value to serialize

        Returns
        -------
        bytes | str
            Serialized value
        """
        if self.format == FileFormat.JSON:
            return json.dumps(value, indent=2)
        if self.format == FileFormat.PICKLE:
            return pickle.dumps(value)
        # TEXT
        return str(value)

    def _deserialize(self, data: bytes | str) -> Any:
        """Deserialize value based on format.

        Parameters
        ----------
        data : bytes | str
            Serialized data

        Returns
        -------
        Any
            Deserialized value
        """
        if self.format == FileFormat.JSON:
            # JSON only accepts str, not bytes
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            return json.loads(data)
        if self.format == FileFormat.PICKLE:
            # Pickle only accepts bytes
            if isinstance(data, str):
                data = data.encode("utf-8")
            # WARNING: Pickle can execute arbitrary code - only use with trusted data
            return pickle.loads(data)  # nosec B301 - User explicitly chose pickle format
        # TEXT
        return data

    async def aget(self, key: str) -> Any:
        """Retrieve a value from file storage.

        Parameters
        ----------
        key : str
            The key to retrieve

        Returns
        -------
        Any
            The stored value, or None if key doesn't exist
        """
        file_path = self._get_file_path(key)

        if not file_path.exists():
            return None

        try:
            data: bytes | str
            if self.format == FileFormat.PICKLE:
                data = file_path.read_bytes()
            else:
                data = file_path.read_text(encoding="utf-8")

            return self._deserialize(data)
        except Exception as e:
            logger.error(f"Failed to read key '{key}' from {file_path}: {e}")
            return None

    async def aset(self, key: str, value: Any) -> None:
        """Store a value in file storage.

        Parameters
        ----------
        key : str
            The key to store under
        value : Any
            The value to store
        """
        file_path = self._get_file_path(key)

        # Ensure parent directory exists
        if self.create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            serialized = self._serialize(value)

            if self.format == FileFormat.PICKLE:
                # Pickle always returns bytes
                if not isinstance(serialized, bytes):
                    raise TypeError(f"Expected bytes for pickle format, got {type(serialized)}")
                file_path.write_bytes(serialized)
            else:
                # JSON and TEXT always return str
                if not isinstance(serialized, str):
                    raise TypeError(
                        f"Expected str for {self.format} format, got {type(serialized)}"
                    )
                file_path.write_text(serialized, encoding="utf-8")

            logger.debug(f"Stored key '{key}' at {file_path}")
        except Exception as e:
            logger.error(f"Failed to write key '{key}' to {file_path}: {e}")
            raise

    async def adelete(self, key: str) -> bool:
        """Delete a key from file storage.

        Parameters
        ----------
        key : str
            The key to delete

        Returns
        -------
        bool
            True if key existed and was deleted, False otherwise
        """
        file_path = self._get_file_path(key)

        if not file_path.exists():
            return False

        try:
            file_path.unlink()
            logger.debug(f"Deleted key '{key}' from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete key '{key}' at {file_path}: {e}")
            return False

    async def alist_keys(self, prefix: str | None = None) -> list[str]:
        """List all keys in storage, optionally filtered by prefix.

        Parameters
        ----------
        prefix : str | None
            Optional prefix to filter keys

        Returns
        -------
        list[str]
            List of matching keys
        """
        if not self.base_path.exists():
            return []

        keys = []
        pattern = f"*.{self.extension}"

        for file_path in self.base_path.glob(pattern):
            # Remove extension and reverse sanitization
            key = file_path.stem

            if prefix is None or key.startswith(prefix):
                keys.append(key)

        return sorted(keys)

    async def aclear(self) -> None:
        """Clear all keys from storage."""
        if not self.base_path.exists():
            return

        pattern = f"*.{self.extension}"
        for file_path in self.base_path.glob(pattern):
            file_path.unlink()

        logger.info(f"Cleared all files from {self.base_path}")

    def __repr__(self) -> str:
        """Return string representation."""
        return f"FileMemoryAdapter(path='{self.base_path}', format='{self.format.value}')"
