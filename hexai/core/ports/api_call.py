"""Base protocol interface for making API calls.

This is a fundamental protocol that other ports can inherit from.
"""

from abc import abstractmethod
from typing import Any, Protocol, runtime_checkable

from hexai.core.registry.decorators import port


@port(
    name="api_call",
    namespace="core",
)
@runtime_checkable
class APICall(Protocol):
    """Base protocol for making external API calls.

    This is a fundamental protocol that provides a standard interface for
    making HTTP/REST API calls. Other protocols (like LLM) can inherit from
    this to indicate they also support API call functionality.

    Implementations can use requests, httpx, aiohttp, or any HTTP client.
    """

    # Required
    @abstractmethod
    async def aget(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make an async GET request.

        Args
        ----
            url: The URL to make the request to
            headers: Optional headers to include
            params: Optional query parameters
            **kwargs: Additional implementation-specific options

        Returns
        -------
            Response data as a dictionary
        """
        ...

    @abstractmethod
    async def apost(
        self,
        url: str,
        json: dict[str, Any] | None = None,
        data: Any | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make an async POST request.

        Args
        ----
            url: The URL to make the request to
            json: JSON data to send in the request body
            data: Form data or raw data to send
            headers: Optional headers to include
            **kwargs: Additional implementation-specific options

        Returns
        -------
            Response data as a dictionary
        """
        ...

    # Optional methods for enhanced functionality
    async def aput(
        self,
        url: str,
        json: dict[str, Any] | None = None,
        data: Any | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make an async PUT request.

        Args
        ----
            url: The URL to make the request to
            json: JSON data to send in the request body
            data: Form data or raw data to send
            headers: Optional headers to include
            **kwargs: Additional implementation-specific options

        Returns
        -------
            Response data as a dictionary
        """
        ...

    async def adelete(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make an async DELETE request.

        Args
        ----
            url: The URL to make the request to
            headers: Optional headers to include
            **kwargs: Additional implementation-specific options

        Returns
        -------
            Response data as a dictionary
        """
        ...

    async def arequest(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make a generic async HTTP request.

        Args
        ----
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            url: The URL to make the request to
            **kwargs: Additional request options (headers, data, json, etc.)

        Returns
        -------
            Response data as a dictionary
        """
        ...
