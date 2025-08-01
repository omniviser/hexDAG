"""Test cases for chat sessions endpoints."""

import pytest
import requests

from ..config import BASE_URL, TEST_TENANT_ID, TEST_USER_ID


def test_get_chat_sessions_success(headers: dict) -> None:
    """Test successful retrieval of chat sessions."""
    url = f"{BASE_URL}/api/chat/sessions"
    params = {"tenant_id": TEST_TENANT_ID, "user_id": TEST_USER_ID}
    response = requests.get(url, headers=headers, params=params, timeout=(3, 10))
    assert response.status_code == 200
    data = response.json()
    sessions = data["sessions"]
    assert isinstance(sessions, list)


@pytest.mark.skip(reason="Auth flow being updated")
def test_get_chat_sessions_unauthorized() -> None:
    """Test chat sessions retrieval without authorization."""
    url = f"{BASE_URL}/api/chat/sessions"
    params = {"tenant_id": TEST_TENANT_ID, "user_id": TEST_USER_ID}
    response = requests.get(url, params=params, timeout=(3, 10))
    assert response.status_code == 401


@pytest.mark.skip(reason="Auth flow being updated")
def test_get_chat_sessions_forbidden(token_with_no_permissions: str) -> None:
    """Test chat sessions retrieval with invalid permissions."""
    headers = {"Authorization": f"Bearer {token_with_no_permissions}"}
    url = f"{BASE_URL}/api/chat/sessions"
    params = {"tenant_id": TEST_TENANT_ID, "user_id": TEST_USER_ID}
    response = requests.get(url, headers=headers, params=params, timeout=(3, 10))
    assert response.status_code == 403


@pytest.mark.skip(reason="Auth flow being updated")
def test_get_chat_sessions_not_found(headers: dict) -> None:
    """Test chat sessions retrieval for non-existent user."""
    url = f"{BASE_URL}/api/chat/sessions"
    params = {"tenant_id": TEST_TENANT_ID, "user_id": "nonexistent"}
    response = requests.get(url, headers=headers, params=params, timeout=(3, 10))
    assert response.status_code == 200  # Returns empty list for non-existent user
    data = response.json()
    assert data["sessions"] == []


@pytest.mark.skip(reason="Auth flow being updated")
def test_get_chat_sessions_validation_error(headers: dict) -> None:
    """Test chat sessions retrieval with invalid parameters."""
    url = f"{BASE_URL}/api/chat/sessions"
    params = {
        "tenant_id": TEST_TENANT_ID,
        "user_id": "",  # Empty user_id should trigger validation error
    }
    response = requests.get(url, headers=headers, params=params, timeout=(3, 10))
    assert response.status_code == 422
