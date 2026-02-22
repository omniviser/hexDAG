"""Mock HTTP client implementation for testing purposes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hexdag.kernel.ports.api_call import APICall


@dataclass
class RecordedRequest:
    """A recorded HTTP request for test assertions."""

    method: str
    url: str
    headers: dict[str, str] | None = None
    params: dict[str, Any] | None = None
    json: dict[str, Any] | None = None
    data: Any | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


class MockHttpClient(APICall):
    """Mock APICall adapter for testing.

    Records all requests and returns pre-configured responses.

    Parameters
    ----------
    responses : dict | list[dict] | None
        Pre-configured responses to cycle through. Each response should
        be a dict with optional keys: ``status_code``, ``headers``, ``body``.
        Defaults to ``{"status_code": 200, "headers": {}, "body": {}}``.
    status_code : int
        Default status code when responses don't specify one (default: 200).

    Examples
    --------
    Basic usage::

        mock = MockHttpClient(responses={"body": {"users": []}})
        result = await mock.aget("https://api.example.com/users")
        assert result["body"] == {"users": []}
        assert len(mock.requests) == 1

    Multiple responses::

        mock = MockHttpClient(responses=[
            {"body": {"id": 1}},
            {"body": {"id": 2}},
        ])
        r1 = await mock.apost("/items", json={"name": "a"})
        r2 = await mock.apost("/items", json={"name": "b"})
        assert r1["body"]["id"] == 1
        assert r2["body"]["id"] == 2
    """

    def __init__(
        self,
        responses: dict[str, Any] | list[dict[str, Any]] | None = None,
        status_code: int = 200,
        **kwargs: Any,
    ) -> None:
        self._default_status_code = status_code
        self.requests: list[RecordedRequest] = []
        self.call_count = 0

        if responses is None:
            self._responses: list[dict[str, Any]] = [
                {"status_code": status_code, "headers": {}, "body": {}}
            ]
        elif isinstance(responses, dict):
            self._responses = [responses]
        else:
            self._responses = list(responses)

    def _get_response(self) -> dict[str, Any]:
        """Get the next response, cycling through the list."""
        if self.call_count < len(self._responses):
            resp = self._responses[self.call_count]
        else:
            resp = self._responses[-1]

        self.call_count += 1

        return {
            "status_code": resp.get("status_code", self._default_status_code),
            "headers": resp.get("headers", {}),
            "body": resp.get("body", {}),
        }

    def _record(
        self,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        data: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Record an HTTP request for later inspection."""
        self.requests.append(
            RecordedRequest(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=json,
                data=data,
                kwargs=kwargs,
            )
        )

    async def aget(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Mock GET request."""
        self._record("GET", url, headers=headers, params=params, **kwargs)
        return self._get_response()

    async def apost(
        self,
        url: str,
        json: dict[str, Any] | None = None,
        data: Any | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Mock POST request."""
        self._record("POST", url, headers=headers, json=json, data=data, **kwargs)
        return self._get_response()

    async def aput(
        self,
        url: str,
        json: dict[str, Any] | None = None,
        data: Any | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Mock PUT request."""
        self._record("PUT", url, headers=headers, json=json, data=data, **kwargs)
        return self._get_response()

    async def adelete(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Mock DELETE request."""
        self._record("DELETE", url, headers=headers, **kwargs)
        return self._get_response()

    async def arequest(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Mock generic request."""
        self._record(method, url, **kwargs)
        return self._get_response()

    def reset(self) -> None:
        """Reset mock state for reuse across tests."""
        self.requests.clear()
        self.call_count = 0
