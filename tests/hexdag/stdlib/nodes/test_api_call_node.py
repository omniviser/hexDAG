"""Tests for ApiCallNode."""

from __future__ import annotations

import pytest

from hexdag.kernel.context import set_ports
from hexdag.stdlib.adapters.mock.mock_http import MockHttpClient
from hexdag.stdlib.nodes.api_call_node import ApiCallNode


class TestApiCallNodeCreation:
    """Test ApiCallNode factory creation."""

    def test_creates_node_spec(self) -> None:
        factory = ApiCallNode()
        spec = factory(name="ping", method="GET", url="https://example.com/health")

        assert spec.name == "ping"
        assert spec.fn is not None
        assert spec.deps == frozenset()

    def test_creates_node_spec_with_deps(self) -> None:
        factory = ApiCallNode()
        spec = factory(
            name="fetch",
            method="GET",
            url="https://example.com/data",
            deps=["auth_node"],
        )

        assert "auth_node" in spec.deps

    def test_function_name_set_correctly(self) -> None:
        factory = ApiCallNode()
        spec = factory(name="my_call", method="POST", url="/api/data")

        assert spec.fn.__name__ == "api_call_my_call"

    def test_infers_input_schema_from_url_template(self) -> None:
        factory = ApiCallNode()
        spec = factory(name="fetch", method="GET", url="/users/{{user_id}}")

        assert spec.in_model is not None
        # The model should have user_id as a field
        assert "user_id" in spec.in_model.model_fields

    def test_infers_input_schema_from_header_templates(self) -> None:
        factory = ApiCallNode()
        spec = factory(
            name="fetch",
            method="GET",
            url="/data",
            headers={"Authorization": "Bearer {{token}}"},
        )

        assert spec.in_model is not None
        assert "token" in spec.in_model.model_fields

    def test_combines_url_and_header_template_vars(self) -> None:
        factory = ApiCallNode()
        spec = factory(
            name="fetch",
            method="GET",
            url="/users/{{user_id}}",
            headers={"Authorization": "Bearer {{token}}"},
        )

        assert spec.in_model is not None
        assert "user_id" in spec.in_model.model_fields
        assert "token" in spec.in_model.model_fields

    def test_no_input_schema_when_no_templates(self) -> None:
        factory = ApiCallNode()
        spec = factory(name="ping", method="GET", url="https://example.com")

        # No template variables → no input model
        assert spec.in_model is None


class TestApiCallNodeExecution:
    """Test ApiCallNode execution with MockHttpClient."""

    @pytest.mark.asyncio
    async def test_get_request(self) -> None:
        mock = MockHttpClient(responses={"body": {"users": [{"id": 1}]}})
        set_ports({"api_call": mock})

        try:
            factory = ApiCallNode()
            spec = factory(name="get_users", method="GET", url="/users")
            result = await spec.fn({})

            assert result["status_code"] == 200
            assert result["body"] == {"users": [{"id": 1}]}
            assert result["error"] is None
            assert len(mock.requests) == 1
            assert mock.requests[0].method == "GET"
            assert mock.requests[0].url == "/users"
        finally:
            set_ports(None)

    @pytest.mark.asyncio
    async def test_post_request_with_json_body(self) -> None:
        mock = MockHttpClient(responses={"body": {"id": 42}})
        set_ports({"api_call": mock})

        try:
            factory = ApiCallNode()
            spec = factory(
                name="create_item",
                method="POST",
                url="/items",
                json_body={"name": "widget", "price": 9.99},
            )
            result = await spec.fn({})

            assert result["body"] == {"id": 42}
            assert mock.requests[0].method == "POST"
            assert mock.requests[0].json == {"name": "widget", "price": 9.99}
        finally:
            set_ports(None)

    @pytest.mark.asyncio
    async def test_put_request(self) -> None:
        mock = MockHttpClient(responses={"body": {"updated": True}})
        set_ports({"api_call": mock})

        try:
            factory = ApiCallNode()
            spec = factory(
                name="update_item",
                method="PUT",
                url="/items/1",
                json_body={"name": "updated"},
            )
            result = await spec.fn({})

            assert result["body"] == {"updated": True}
            assert mock.requests[0].method == "PUT"
        finally:
            set_ports(None)

    @pytest.mark.asyncio
    async def test_delete_request(self) -> None:
        mock = MockHttpClient(responses={"body": {"deleted": True}})
        set_ports({"api_call": mock})

        try:
            factory = ApiCallNode()
            spec = factory(name="delete_item", method="DELETE", url="/items/1")
            result = await spec.fn({})

            assert result["body"] == {"deleted": True}
            assert mock.requests[0].method == "DELETE"
        finally:
            set_ports(None)

    @pytest.mark.asyncio
    async def test_patch_request(self) -> None:
        mock = MockHttpClient(responses={"body": {"patched": True}})
        set_ports({"api_call": mock})

        try:
            factory = ApiCallNode()
            spec = factory(
                name="patch_item",
                method="PATCH",
                url="/items/1",
                json_body={"name": "patched"},
            )
            result = await spec.fn({})

            assert result["body"] == {"patched": True}
            assert mock.requests[0].method == "PATCH"
        finally:
            set_ports(None)

    @pytest.mark.asyncio
    async def test_url_templating(self) -> None:
        mock = MockHttpClient(responses={"body": {"name": "Alice"}})
        set_ports({"api_call": mock})

        try:
            factory = ApiCallNode()
            spec = factory(
                name="get_user",
                method="GET",
                url="/users/{{user_id}}/profile",
            )
            result = await spec.fn({"user_id": "42"})

            assert result["body"] == {"name": "Alice"}
            assert mock.requests[0].url == "/users/42/profile"
        finally:
            set_ports(None)

    @pytest.mark.asyncio
    async def test_header_templating(self) -> None:
        mock = MockHttpClient(responses={"body": {}})
        set_ports({"api_call": mock})

        try:
            factory = ApiCallNode()
            spec = factory(
                name="authed_call",
                method="GET",
                url="/data",
                headers={"Authorization": "Bearer {{token}}"},
            )
            result = await spec.fn({"token": "my-secret-token"})

            assert result["error"] is None
            assert mock.requests[0].headers == {"Authorization": "Bearer my-secret-token"}
        finally:
            set_ports(None)

    @pytest.mark.asyncio
    async def test_query_params_passed(self) -> None:
        mock = MockHttpClient(responses={"body": []})
        set_ports({"api_call": mock})

        try:
            factory = ApiCallNode()
            spec = factory(
                name="search",
                method="GET",
                url="/search",
                params={"q": "test", "limit": "10"},
            )
            result = await spec.fn({})

            assert result["error"] is None
            assert mock.requests[0].params == {"q": "test", "limit": "10"}
        finally:
            set_ports(None)

    @pytest.mark.asyncio
    async def test_auto_creates_driver_when_no_port(self) -> None:
        """When no api_call port is configured, a default HttpClientDriver is created."""
        set_ports({})  # No api_call port

        try:
            factory = ApiCallNode()
            spec = factory(name="auto_call", method="GET", url="https://httpbin.org/get")

            # The function should be callable (we can't actually make HTTP calls in tests,
            # but we verify it doesn't fail at creation time)
            assert spec.fn is not None
            assert spec.fn.__name__ == "api_call_auto_call"
        finally:
            set_ports(None)

    @pytest.mark.asyncio
    async def test_custom_port_name(self) -> None:
        mock = MockHttpClient(responses={"body": {"ok": True}})
        set_ports({"my_api": mock})

        try:
            factory = ApiCallNode()
            spec = factory(
                name="custom",
                method="GET",
                url="/check",
                port="my_api",
            )
            result = await spec.fn({})

            assert result["body"] == {"ok": True}
        finally:
            set_ports(None)

    @pytest.mark.asyncio
    async def test_static_headers_without_templates(self) -> None:
        mock = MockHttpClient(responses={"body": {}})
        set_ports({"api_call": mock})

        try:
            factory = ApiCallNode()
            spec = factory(
                name="static_headers",
                method="GET",
                url="/data",
                headers={"X-Custom": "static-value"},
            )
            result = await spec.fn({})

            assert result["error"] is None
            assert mock.requests[0].headers == {"X-Custom": "static-value"}
        finally:
            set_ports(None)


class TestApiCallNodeTopLevelExport:
    """Test that HttpClientDriver and MockHttpClient are importable from hexdag."""

    def test_http_client_driver_importable(self) -> None:
        from hexdag import HttpClientDriver

        assert HttpClientDriver is not None

    def test_mock_http_client_importable(self) -> None:
        from hexdag import MockHttpClient

        assert MockHttpClient is not None

    def test_api_call_node_importable_from_stdlib(self) -> None:
        from hexdag.stdlib.nodes import ApiCallNode

        assert ApiCallNode is not None
