"""Test suite for the / endpoint.

This module tests authorization header verification dependency.
"""

import requests

from ..config import BASE_URL


def _requests_get_to_secured_endpoint(headers: dict | None) -> requests.Response:
    """Helper function to make GET requests to the secured endpoint."""
    test_url = f"{BASE_URL}/secured/"
    response = requests.get(test_url, params={"delay_time": 5.0}, timeout=(3, 10), headers=headers)
    return response

def test_auth_secured_endpoint_successfully(headers: dict) -> None:
    """Test that the secured endpoint returns a 200 status code with valid headers."""
    response = _requests_get_to_secured_endpoint(headers)
    assert response.status_code == 200

def test_missing_authorization_header():
    response = _requests_get_to_secured_endpoint(None)
    assert response.status_code == 400
    assert response.json() == {"error": "Missing Authorization header"}

def test_invalid_authorization_format():
    response = _requests_get_to_secured_endpoint({"Authorization": "invalid"})
    assert response.status_code == 400
    assert response.json() == {"error": "Invalid Authorization header format"}

def test_invalid_auth_key():
    response = _requests_get_to_secured_endpoint({"Authorization": "Bearer invalid"})
    assert response.status_code == 403
    assert response.json() == {"error": "Invalid or missing API key"}
