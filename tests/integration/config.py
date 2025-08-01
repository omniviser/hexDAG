"""Configuration settings for integration tests.

This module contains configuration variables used across the test suite.
"""

FAST_API_PORT = "8001"
DJANGO_PORT = "8000"

# Base URL for the FastAPI application
BASE_URL = f"http://localhost:{FAST_API_PORT}"

# Base URL for Django authentication
AUTH_URL = f"https://omniviser.local:{DJANGO_PORT}"

# Test tenant and user IDs for API calls
TEST_TENANT_ID = "test-tenant"
TEST_USER_ID = "test-user"
