"""API extraction node for HTTP/REST API data extraction."""

from typing import Any, Literal

from hexdag.kernel.domain.dag import NodeSpec

from .base_node_factory import BaseNodeFactory

# Convention: HTTP methods for dropdown menus in Studio UI
HttpMethod = Literal["GET", "POST", "PUT", "DELETE", "PATCH"]


class APIExtractNode(BaseNodeFactory):
    """Extract data from REST APIs with pagination, authentication, and error handling.

    Supports:
    - GET/POST requests
    - Bearer token, API key, OAuth authentication
    - Pagination (cursor, offset, page)
    - Rate limiting
    - Retry logic
    - Structured error handling

    Examples
    --------
    YAML pipeline::

        - kind: api_extract_node
          metadata:
            name: fetch_customers
          spec:
            endpoint: https://api.example.com/v1/customers
            method: GET
            params:
              limit: 100
              status: active
            pagination:
              type: cursor
              cursor_param: after
              cursor_path: meta.next_cursor
              has_more_path: meta.has_more
            auth:
              type: bearer
              token: ${API_TOKEN}
            output_artifact:
              slot: raw_customers
              key: customers_2024_01_15
              format: json
    """

    # Studio UI metadata
    _hexdag_icon = "Globe"
    _hexdag_color = "#06b6d4"  # cyan-500

    def __call__(
        self,
        name: str,
        endpoint: str,
        method: HttpMethod = "GET",
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        auth: dict[str, Any] | None = None,
        pagination: dict[str, Any] | None = None,
        output_artifact: dict[str, Any] | None = None,
        timeout: int = 30,
        max_retries: int = 3,
        backoff_factor: float = 0.3,
        rate_limit: dict[str, Any] | None = None,
        deps: list[str] | None = None,
        **kwargs: Any,
    ) -> NodeSpec:
        """Create API extraction node specification.

        Parameters
        ----------
        name : str
            Node name
        endpoint : str
            API endpoint URL
        method : HttpMethod
            HTTP method: 'GET', 'POST', 'PUT', 'DELETE', 'PATCH'
        params : dict, optional
            Query parameters or request body
        headers : dict, optional
            Additional HTTP headers
        auth : dict, optional
            Authentication configuration
        pagination : dict, optional
            Pagination configuration
        output_artifact : dict, optional
            Output artifact configuration
        timeout : int
            Request timeout in seconds
        max_retries : int
            Maximum retry attempts
        backoff_factor : float
            Backoff multiplier for retries
        rate_limit : dict, optional
            Rate limiting configuration
        deps : list[str], optional
            Dependency node names
        **kwargs : Any
            Additional parameters

        Returns
        -------
        NodeSpec
            Node specification ready for execution
        """
        # Create wrapped function
        wrapped_fn = self._create_api_function(
            name,
            endpoint,
            method,
            params,
            headers,
            auth,
            pagination,
            output_artifact,
            timeout,
            max_retries,
            backoff_factor,
            rate_limit,
        )

        # Define schemas
        input_schema = {"input_data": dict, "**ports": dict}
        output_schema = {"output": dict, "metadata": dict}

        input_model = self.create_pydantic_model(f"{name}Input", input_schema)
        output_model = self.create_pydantic_model(f"{name}Output", output_schema)

        # Store parameters
        node_params = {
            "endpoint": endpoint,
            "method": method,
            "params": params,
            "headers": headers,
            "auth": auth,
            "pagination": pagination,
            "output_artifact": output_artifact,
            "timeout": timeout,
            "max_retries": max_retries,
            "backoff_factor": backoff_factor,
            "rate_limit": rate_limit,
            **kwargs,
        }

        return NodeSpec(
            name=name,
            fn=wrapped_fn,
            in_model=input_model,
            out_model=output_model,
            deps=frozenset(deps or []),
            params=node_params,
        )

    def _create_api_function(
        self,
        name: str,
        endpoint: str,
        method: str,
        params: dict[str, Any] | None,
        headers: dict[str, str] | None,
        auth: dict[str, Any] | None,
        pagination: dict[str, Any] | None,
        output_artifact: dict[str, Any] | None,
        timeout: int,
        max_retries: int,
        backoff_factor: float,
        rate_limit: dict[str, Any] | None,
    ) -> Any:
        """Create the wrapped API extraction function.

        Implementation details omitted for brevity - similar to the original
        but simplified for demo purposes.

        Parameters
        ----------
        name : str
            Node name
        endpoint : str
            API endpoint
        method : str
            HTTP method
        params : dict, optional
            Request parameters
        headers : dict, optional
            Request headers
        auth : dict, optional
            Authentication config
        pagination : dict, optional
            Pagination config
        output_artifact : dict, optional
            Output artifact config
        timeout : int
            Request timeout
        max_retries : int
            Maximum retries
        backoff_factor : float
            Backoff factor
        rate_limit : dict, optional
            Rate limiting config

        Returns
        -------
        Callable
            Async function that performs API extraction
        """

        # Implementation would go here - simplified for the example
        async def wrapped_fn(input_data: Any, **ports: Any) -> dict[str, Any]:
            """Placeholder implementation."""
            # In a real implementation, this would:
            # 1. Make HTTP requests with aiohttp
            # 2. Handle pagination
            # 3. Apply authentication
            # 4. Store results in artifact store
            # 5. Return structured results

            return {
                "data": [],
                "metadata": {
                    "endpoint": endpoint,
                    "method": method,
                    "records_extracted": 0,
                    "status": "placeholder_implementation",
                },
            }

        wrapped_fn.__name__ = f"api_extract_{name}"
        wrapped_fn.__doc__ = f"API extraction: {endpoint}"

        return wrapped_fn
