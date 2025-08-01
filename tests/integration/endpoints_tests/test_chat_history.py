"""Test cases for chat history endpoints."""

import pytest
import requests

from ..config import BASE_URL, TEST_TENANT_ID, TEST_USER_ID


@pytest.mark.skip(reason="No user inserted")
def test_get_chat_history_success(headers: dict) -> None:
    """Test successful retrieval of chat history."""
    url = f"{BASE_URL}/api/chat/history/tenant_id={TEST_TENANT_ID}&user_id={TEST_USER_ID}"
    response = requests.get(url, headers=headers, timeout=(3, 10))
    assert response.status_code == 200


@pytest.mark.skip(reason="Auth flow being updated")
def test_get_chat_history_unauthorized() -> None:
    """Test chat history retrieval without authorization."""
    url = f"{BASE_URL}/api/chat/history/tenant_id={TEST_TENANT_ID}&user_id={TEST_USER_ID}"
    response = requests.get(url, timeout=(3, 10))
    assert response.status_code == 401


@pytest.mark.skip(reason="Auth flow being updated")
def test_get_chat_history_forbidden(headers: dict) -> None:
    """Test chat history retrieval with invalid permissions."""
    url = f"{BASE_URL}/api/chat/history/tenant_id={TEST_TENANT_ID}&user_id={TEST_USER_ID}"
    response = requests.get(url, headers=headers, timeout=(3, 10))
    assert response.status_code == 403


def test_get_chat_history_not_found(headers: dict) -> None:
    """Test chat history retrieval for non-existent session."""
    url = f"{BASE_URL}/api/chat/history/tenant_id={TEST_TENANT_ID}&user_id={TEST_USER_ID}"
    response = requests.get(url, headers=headers, timeout=(3, 10))
    assert response.status_code == 404  # MongoDB error when session not found
    data = response.json()
    assert data["detail"] == "Not Found"


@pytest.mark.skip(reason="Auth flow being updated")
def test_get_chat_history_validation_error(headers: dict) -> None:
    """Test chat history retrieval with invalid parameters."""
    url = f"{BASE_URL}/api/chat/history/tenant_id={TEST_TENANT_ID}&user_id="  # Empty user_id should trigger validation error
    response = requests.get(url, headers=headers, timeout=(3, 10))
    assert response.status_code == 422
