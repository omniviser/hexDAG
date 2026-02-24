"""Example: ApiCallNode — declarative HTTP calls from YAML pipelines.

Demonstrates how to use the ApiCallNode with HttpClientDriver for REST API
integration, and MockHttpClient for testing.
"""

import asyncio

from hexdag.kernel.context import set_ports
from hexdag.stdlib.adapters.mock.mock_http import MockHttpClient
from hexdag.stdlib.nodes.api_call_node import ApiCallNode


async def main() -> None:
    # --- 1. Basic GET request ---
    mock = MockHttpClient(responses={"body": {"users": [{"id": 1, "name": "Alice"}]}})
    set_ports({"api_call": mock})

    factory = ApiCallNode()
    node = factory(name="list_users", method="GET", url="/api/users", params={"limit": "10"})

    result = await node.fn({})
    print(f"GET /api/users → {result['body']}")

    # --- 2. POST with JSON body ---
    mock = MockHttpClient(responses={"body": {"id": 42, "status": "created"}})
    set_ports({"api_call": mock})

    node = factory(
        name="create_order",
        method="POST",
        url="/api/orders",
        json_body={"customer": "Alice", "items": ["widget"]},
    )

    result = await node.fn({})
    print(f"POST /api/orders → {result['body']}")

    # --- 3. URL templating ---
    mock = MockHttpClient(responses={"body": {"name": "Alice", "email": "alice@example.com"}})
    set_ports({"api_call": mock})

    node = factory(
        name="get_user",
        method="GET",
        url="/api/users/{{user_id}}/profile",
        headers={"Authorization": "Bearer {{token}}"},
    )

    result = await node.fn({"user_id": "42", "token": "my-secret"})
    print(f"GET /api/users/42/profile → {result['body']}")
    print(f"  Recorded URL: {mock.requests[0].url}")
    print(f"  Recorded headers: {mock.requests[0].headers}")

    set_ports(None)

    # --- 4. YAML pipeline equivalent ---
    print(
        """
YAML equivalent:

    spec:
      ports:
        api_call:
          adapter: hexdag.drivers.http_client.HttpClientDriver
          config:
            base_url: "https://api.example.com"
            timeout: 30.0
            headers:
              X-API-Key: "${API_KEY}"

      nodes:
        - kind: api_call_node
          metadata:
            name: get_user
          spec:
            method: GET
            url: "/users/{{user_id}}/profile"
            headers:
              Authorization: "Bearer {{token}}"
          dependencies: []
"""
    )


if __name__ == "__main__":
    asyncio.run(main())
