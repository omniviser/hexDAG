"""Configuration file for pytest containing fixtures and configuration.

This module provides fixtures that can be used across multiple test files:
- token: Provides a JWT token for an authorized user
"""

import pytest


@pytest.fixture(scope="session")
def headers() -> dict[str, str]:
    """Fixture that creates a mock JWT token for testing."""
    return {"Authorization": f"Bearer TEST_FASTAPI_KEY"}
