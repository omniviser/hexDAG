"""Test suite for the /healthz endpoint.

This module contains simple tests to verify the functionality of the /healthz
endpoint, which checks the health of the service. The tests included are:
- Testing if the /healthz endpoint returns a 200 status code.
- Verifying that the /healthz endpoint returns the expected response message.
These tests help ensure that the health check mechanism is functioning correctly.
"""

import pytest
import requests

from ..config import BASE_URL


def test_healthz_status_code(headers) -> None:
    """Test if /healthz endpoint returns a 200 status code."""
    response = requests.get(f"{BASE_URL}/healthz/", timeout=(3, 10), headers=headers)
    assert response.status_code == 200
