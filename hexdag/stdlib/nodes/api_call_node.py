"""ApiCallNode - Declarative HTTP calls from YAML pipelines.

Provides ergonomic REST API access via the ``api_call`` port, with
URL/header/body templating using Jinja2-style ``{{variable}}`` syntax.

Examples
--------
YAML pipeline usage::

    - kind: api_call_node
      metadata:
        name: fetch_users
      spec:
        method: GET
        url: "/users/{{user_id}}"
        headers:
          Authorization: "Bearer {{api_token}}"
        params:
          limit: "10"
      dependencies: []

Python usage::

    from hexdag.stdlib.nodes import ApiCallNode

    factory = ApiCallNode()
    node = factory(
        name="fetch_users",
        method="GET",
        url="https://api.example.com/users",
        params={"limit": "10"},
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

from hexdag.drivers.http_client import HttpClientDriver  # lazy: avoid circular import
from hexdag.kernel.context import get_port
from hexdag.kernel.logging import get_logger
from hexdag.kernel.orchestration.prompt.template import PromptTemplate
from hexdag.kernel.ports.api_call import APICall  # noqa: TC001
from hexdag.kernel.protocols import to_dict
from hexdag.kernel.utils.node_timer import node_timer

from .base_node_factory import BaseNodeFactory

if TYPE_CHECKING:
    from pydantic import BaseModel

    from hexdag.kernel.domain.dag import NodeSpec

logger = get_logger(__name__)

# Convention: HTTP method options for dropdown menus in Studio UI
HttpMethod = Literal["GET", "POST", "PUT", "DELETE", "PATCH"]

# Maps HTTP method names to APICall port method names
_METHOD_MAP: dict[str, str] = {
    "GET": "aget",
    "POST": "apost",
    "PUT": "aput",
    "DELETE": "adelete",
    "PATCH": "arequest",  # PATCH uses generic arequest with method param
}


class ApiCallNode(BaseNodeFactory, yaml_alias="api_call_node"):
    """Declarative HTTP call node for REST API integration.

    Makes HTTP requests via the ``api_call`` port with URL, header, and body
    templating. If no ``api_call`` port is configured, a default
    ``HttpClientDriver`` is created automatically.

    Port Capabilities
    -----------------
    Uses ``api_call`` port implementing ``APICall`` (auto-created if missing).

    Examples
    --------
    Simple GET request::

        factory = ApiCallNode()
        node = factory(
            name="get_users",
            method="GET",
            url="https://api.example.com/users",
        )

    POST with templated body::

        node = factory(
            name="create_order",
            method="POST",
            url="https://api.example.com/orders",
            json_body={"customer": "{{customer_id}}", "items": "{{items}}"},
        )

    YAML pipeline::

        - kind: api_call_node
          metadata:
            name: fetch_data
          spec:
            method: GET
            url: "https://api.example.com/data/{{id}}"
            headers:
              Authorization: "Bearer {{token}}"
          dependencies: []
    """

    # Port capability table (validated at mount time by orchestrator)
    _hexdag_port_capabilities: ClassVar[dict[str, list[type]]] = {
        "api_call": [APICall],
    }

    # Studio UI metadata
    _hexdag_icon = "Globe"
    _hexdag_color = "#06b6d4"  # cyan-500

    def __call__(
        self,
        name: str,
        method: HttpMethod = "GET",
        url: str = "",
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        port: str = "api_call",
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        deps: list[str] | None = None,
        **kwargs: Any,
    ) -> NodeSpec:
        """Create an HTTP call node specification.

        Parameters
        ----------
        name : str
            Node name (must be unique in the pipeline).
        method : str
            HTTP method: GET, POST, PUT, DELETE, or PATCH.
        url : str
            URL or path template. Supports ``{{variable}}`` syntax.
        headers : dict[str, str] | None
            Request headers. Values support ``{{variable}}`` syntax.
        params : dict[str, Any] | None
            Query parameters for GET requests.
        json_body : dict[str, Any] | None
            JSON body for POST/PUT/PATCH requests.
        port : str
            Port name to use (default: ``"api_call"``).
        output_schema : dict | type[BaseModel] | None
            Optional schema for output validation.
        deps : list[str] | None
            Dependency node names.
        **kwargs : Any
            Additional NodeSpec parameters (timeout, max_retries, when, etc.).

        Returns
        -------
        NodeSpec
            Complete node specification ready for execution.

        Examples
        --------
        >>> factory = ApiCallNode()
        >>> node = factory(name="ping", method="GET", url="https://httpbin.org/get")
        >>> node.name
        'ping'
        """
        # Collect all template strings for input schema inference
        templates: list[str] = [url] if url else []
        if headers:
            templates.extend(headers.values())

        # Infer input schema from all template variables
        all_vars: set[str] = set()
        for tmpl_str in templates:
            tmpl = PromptTemplate(tmpl_str)
            all_vars.update(tmpl.input_vars)

        input_schema: dict[str, Any] = dict.fromkeys(sorted(all_vars), str) or None  # type: ignore[assignment]

        # Create the HTTP call wrapper function
        http_wrapper = self._create_http_wrapper(
            name=name,
            method=method,
            url_template=url,
            headers_template=headers,
            params=params,
            json_body=json_body,
            port_name=port,
        )

        return self.create_node_with_mapping(
            name=name,
            wrapped_fn=http_wrapper,
            input_schema=input_schema,
            output_schema=output_schema,
            deps=deps,
            **kwargs,
        )

    def _create_http_wrapper(
        self,
        name: str,
        method: str,
        url_template: str,
        headers_template: dict[str, str] | None,
        params: dict[str, Any] | None,
        json_body: dict[str, Any] | None,
        port_name: str,
    ) -> Any:
        """Create the async wrapper function for HTTP execution."""

        # Pre-compile templates
        url_tmpl = PromptTemplate(url_template) if url_template else None
        header_tmpls: dict[str, PromptTemplate] | None = None
        if headers_template:
            header_tmpls = {k: PromptTemplate(v) for k, v in headers_template.items()}

        # Copy port capabilities metadata
        self._copy_port_capabilities_to_wrapper_for_http(port_name)

        async def http_call_fn(input_data: dict[str, Any]) -> dict[str, Any]:
            """Execute HTTP request."""
            node_logger = logger.bind(
                node=name,
                node_type="api_call_node",
                method=method,
            )

            # Resolve input to dict
            try:
                input_dict = to_dict(input_data)
            except TypeError:
                input_dict = input_data

            # Render URL template
            rendered_url = url_template
            if url_tmpl and url_tmpl.input_vars:
                rendered_url = url_tmpl.render(**input_dict)

            # Render header templates
            rendered_headers: dict[str, str] | None = None
            if header_tmpls:
                rendered_headers = {
                    k: tmpl.render(**input_dict) if tmpl.input_vars else tmpl.template
                    for k, tmpl in header_tmpls.items()
                }

            # Get the API call port (auto-create default if missing)
            api_client = get_port(port_name)
            if api_client is None:
                node_logger.info(
                    "No '{}' port configured, using default HttpClientDriver", port_name
                )
                api_client = HttpClientDriver()

            node_logger.info(
                "HTTP request",
                url=rendered_url,
                has_headers=rendered_headers is not None,
                has_params=params is not None,
                has_body=json_body is not None,
            )

            try:
                with node_timer() as t:
                    result = await _dispatch_request(
                        client=api_client,
                        method=method,
                        url=rendered_url,
                        headers=rendered_headers,
                        params=params,
                        json_body=json_body,
                    )

                node_logger.info(
                    "HTTP response",
                    status_code=result.get("status_code"),
                    duration_ms=t.duration_str,
                )

                return {
                    "status_code": result.get("status_code"),
                    "headers": result.get("headers", {}),
                    "body": result.get("body"),
                    "error": None,
                }

            except Exception as e:
                node_logger.error(
                    "HTTP request failed",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise

        # Preserve function metadata for debugging
        http_call_fn.__name__ = f"api_call_{name}"
        http_call_fn.__doc__ = f"HTTP {method} call: {url_template}"

        return http_call_fn

    def _copy_port_capabilities_to_wrapper_for_http(self, port_name: str) -> None:
        """Set port capabilities metadata for the given port name."""
        # Only set if using default api_call port (matches class-level capabilities)
        if port_name == "api_call":
            return  # Use class-level _hexdag_port_capabilities
        # For custom port names, we can't validate at mount time


async def _dispatch_request(
    client: APICall,
    method: str,
    url: str,
    headers: dict[str, str] | None,
    params: dict[str, Any] | None,
    json_body: dict[str, Any] | None,
) -> dict[str, Any]:
    """Dispatch HTTP request to the appropriate client method."""
    method_upper = method.upper()
    port_method = _METHOD_MAP.get(method_upper)
    if not port_method:
        raise ValueError(f"Unsupported HTTP method: {method}")

    if method_upper == "GET":
        return await client.aget(url, headers=headers, params=params)
    if method_upper == "POST":
        return await client.apost(url, json=json_body, headers=headers)
    if method_upper == "PUT":
        return await client.aput(url, json=json_body, headers=headers)
    if method_upper == "DELETE":
        return await client.adelete(url, headers=headers)
    if method_upper == "PATCH":
        return await client.arequest("PATCH", url, json=json_body, headers=headers)
    raise ValueError(f"Unsupported HTTP method: {method}")
