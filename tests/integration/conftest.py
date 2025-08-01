"""Configuration file for pytest containing fixtures and configuration.

This module provides fixtures that can be used across multiple test files:
- token: Provides a JWT token for an authorized user
"""
from typing import Any, Generator

import pytest


@pytest.fixture(scope="session")
def headers() -> dict[str, str]:
    """Fixture that creates a mock JWT token for testing."""
    return {"Authorization": f"Bearer TEST_FASTAPI_KEY"}


@pytest.fixture(autouse=True)
def reset_singletons() -> Generator[None, Any, None]:
    """Reset singleton instances before each test."""
    from app.services.mongodb_service import MongoDBService
    from app.utils.redis_utils.redis_class import RedisUtils

    RedisUtils.reset_instance()
    MongoDBService.reset_instances()
    yield
