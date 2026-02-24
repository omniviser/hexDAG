"""HTTP client driver using httpx.AsyncClient.

This driver implements the :class:`~hexdag.kernel.ports.api_call.APICall`
protocol, providing async HTTP calls with connection pooling, timeout
configuration, and automatic JSON parsing.
"""

from __future__ import annotations

from typing import Any

import httpx

from hexdag.kernel.exceptions import HttpClientError  # noqa: F401
from hexdag.kernel.logging import get_logger

logger = get_logger(__name__)


class HttpClientDriver:
    """APICall driver using httpx.AsyncClient.

    Provides async HTTP calls with connection pooling, configurable
    timeouts, default headers, and automatic JSON response parsing.

    Parameters
    ----------
    base_url : str
        Optional base URL prefix for all requests.
    timeout : float
        Request timeout in seconds (default: 30.0).
    headers : dict[str, str] | None
        Default headers included in every request.
    bearer_token : str | None
        Bearer token for Authorization header. Adds
        ``Authorization: Bearer <token>`` to default headers.
    basic_auth_username : str | None
        Username for HTTP Basic Auth. Must be paired with
        ``basic_auth_password``.
    basic_auth_password : str | None
        Password for HTTP Basic Auth. Must be paired with
        ``basic_auth_username``.
    follow_redirects : bool
        Whether to follow HTTP redirects (default: True).
    raise_for_status : bool
        If True, raise :class:`HttpClientError` on non-2xx responses
        (default: True).

    Examples
    --------
    Basic usage::

        http = HttpClientDriver(base_url="https://api.example.com")
        result = await http.aget("/users", params={"limit": "10"})
        print(result["body"])

    With bearer token::

        http = HttpClientDriver(
            base_url="https://api.example.com",
            bearer_token="sk-my-token",
        )
        result = await http.apost("/orders", json={"item": "widget"})

    With basic auth::

        http = HttpClientDriver(
            base_url="https://api.example.com",
            basic_auth_username="user",
            basic_auth_password="pass",
        )

    From YAML pipeline::

        ports:
          api_call:
            adapter: hexdag.drivers.http_client.HttpClientDriver
            config:
              base_url: "https://api.example.com"
              timeout: 60.0
              bearer_token: "${API_TOKEN}"
    """

    def __init__(
        self,
        base_url: str = "",
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
        bearer_token: str | None = None,
        basic_auth_username: str | None = None,
        basic_auth_password: str | None = None,
        follow_redirects: bool = True,
        raise_for_status: bool = True,
        **kwargs: Any,
    ) -> None:
        self._base_url = base_url
        self._timeout = timeout
        self._default_headers = dict(headers) if headers else {}
        self._follow_redirects = follow_redirects
        self._raise_for_status = raise_for_status
        self._client: httpx.AsyncClient | None = None
        # Hook for testing — inject a custom transport
        self._transport: httpx.AsyncBaseTransport | None = None

        # Auth: bearer token adds Authorization header
        if bearer_token:
            self._default_headers["Authorization"] = f"Bearer {bearer_token}"

        # Auth: basic auth via httpx.BasicAuth
        self._auth: httpx.BasicAuth | None = None
        if basic_auth_username and basic_auth_password:
            self._auth = httpx.BasicAuth(basic_auth_username, basic_auth_password)

    def _get_client(self) -> httpx.AsyncClient:
        """Lazily create the httpx client on first use."""
        if self._client is None:
            kwargs: dict[str, Any] = {
                "base_url": self._base_url,
                "timeout": self._timeout,
                "headers": self._default_headers,
                "follow_redirects": self._follow_redirects,
            }
            if self._auth is not None:
                kwargs["auth"] = self._auth
            if self._transport is not None:
                kwargs["transport"] = self._transport
            self._client = httpx.AsyncClient(**kwargs)
        return self._client

    def _merge_headers(self, headers: dict[str, str] | None) -> dict[str, str] | None:
        """Merge per-request headers with defaults."""
        if not headers:
            return None
        # Per-request headers override defaults
        return {**self._default_headers, **headers}

    def _parse_response(self, response: httpx.Response) -> dict[str, Any]:
        """Parse an httpx response into a standard dict.

        Returns
        -------
        dict[str, Any]
            ``{"status_code": int, "headers": dict, "body": Any}``
            where body is parsed JSON if content-type is JSON, else raw text.
        """
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                body = response.json()
            except Exception:
                body = response.text
        else:
            body = response.text

        return {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body": body,
        }

    def _check_status(self, result: dict[str, Any]) -> dict[str, Any]:
        """Raise HttpClientError if status is non-2xx and raise_for_status is enabled."""
        if self._raise_for_status:
            status = result["status_code"]
            if status < 200 or status >= 300:
                raise HttpClientError(
                    status_code=status,
                    body=result["body"],
                    message=f"HTTP {status}: {result['body']}",
                )
        return result

    async def aget(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make an async GET request.

        Parameters
        ----------
        url : str
            The URL (or path if base_url is set).
        headers : dict[str, str] | None
            Optional per-request headers.
        params : dict[str, Any] | None
            Optional query parameters.

        Returns
        -------
        dict[str, Any]
            ``{"status_code": int, "headers": dict, "body": Any}``
        """
        client = self._get_client()
        response = await client.get(
            url, headers=self._merge_headers(headers), params=params, **kwargs
        )
        result = self._parse_response(response)
        return self._check_status(result)

    async def apost(
        self,
        url: str,
        json: dict[str, Any] | None = None,
        data: Any | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make an async POST request.

        Parameters
        ----------
        url : str
            The URL (or path if base_url is set).
        json : dict[str, Any] | None
            JSON data for the request body.
        data : Any | None
            Form data or raw data.
        headers : dict[str, str] | None
            Optional per-request headers.

        Returns
        -------
        dict[str, Any]
            ``{"status_code": int, "headers": dict, "body": Any}``
        """
        client = self._get_client()
        response = await client.post(
            url,
            json=json,
            data=data,
            headers=self._merge_headers(headers),
            **kwargs,
        )
        result = self._parse_response(response)
        return self._check_status(result)

    async def aput(
        self,
        url: str,
        json: dict[str, Any] | None = None,
        data: Any | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make an async PUT request.

        Parameters
        ----------
        url : str
            The URL (or path if base_url is set).
        json : dict[str, Any] | None
            JSON data for the request body.
        data : Any | None
            Form data or raw data.
        headers : dict[str, str] | None
            Optional per-request headers.

        Returns
        -------
        dict[str, Any]
            ``{"status_code": int, "headers": dict, "body": Any}``
        """
        client = self._get_client()
        response = await client.put(
            url,
            json=json,
            data=data,
            headers=self._merge_headers(headers),
            **kwargs,
        )
        result = self._parse_response(response)
        return self._check_status(result)

    async def adelete(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make an async DELETE request.

        Parameters
        ----------
        url : str
            The URL (or path if base_url is set).
        headers : dict[str, str] | None
            Optional per-request headers.

        Returns
        -------
        dict[str, Any]
            ``{"status_code": int, "headers": dict, "body": Any}``
        """
        client = self._get_client()
        response = await client.delete(url, headers=self._merge_headers(headers), **kwargs)
        result = self._parse_response(response)
        return self._check_status(result)

    async def arequest(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make a generic async HTTP request.

        Parameters
        ----------
        method : str
            HTTP method (GET, POST, PUT, DELETE, PATCH, etc.).
        url : str
            The URL (or path if base_url is set).
        **kwargs : Any
            Additional request options (headers, json, data, params, etc.).

        Returns
        -------
        dict[str, Any]
            ``{"status_code": int, "headers": dict, "body": Any}``
        """
        if "headers" in kwargs:
            kwargs["headers"] = self._merge_headers(kwargs["headers"])
        client = self._get_client()
        response = await client.request(method, url, **kwargs)
        result = self._parse_response(response)
        return self._check_status(result)

    async def aclose(self) -> None:
        """Close the underlying httpx client and release connection pool resources."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def ahealth_check(self) -> dict[str, Any]:
        """Basic health check — verifies the client can be created."""
        _ = self._get_client()
        return {
            "status": "healthy",
            "adapter_name": "HttpClientDriver",
            "base_url": self._base_url,
        }
