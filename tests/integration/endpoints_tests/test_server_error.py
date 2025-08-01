"""Test cases for server error handling."""

import pytest
import requests

from ..config import BASE_URL, TEST_TENANT_ID, TEST_USER_ID


def test_get_chat_history_server_error(headers: dict) -> None:
    """Test if chat history endpoint handles internal server errors."""
    url = f"{BASE_URL}/api/chat/sessions/nonexistent/messages"
    params = {"tenant_id": TEST_TENANT_ID, "user_id": TEST_USER_ID}
    response = requests.get(url, headers=headers, params=params, timeout=(3, 10))
    assert response.status_code == 500  # MongoDB error when session not found
    data = response.json()
    assert "error" in data


@pytest.mark.skip(reason="Wrong!!!")
def test_get_chat_sessions_server_error(headers: dict) -> None:
    """Test if chat sessions endpoint handles internal server errors."""
    url = f"{BASE_URL}/api/chat/sessions"
    params = {"tenant_id": TEST_TENANT_ID, "user_id": "nonexistent"}
    response = requests.get(url, headers=headers, params=params, timeout=(3, 10))
    assert response.status_code == 200  # Returns empty list for non-existent user


def test_get_task_server_error(headers: dict) -> None:
    """Test if /task endpoint handles validation errors."""
    params = {"delay_time": -1.0}  # Invalid delay time triggers validation error
    url = f"{BASE_URL}/task"
    response = requests.get(url, params=params, timeout=(3, 10), headers=headers)
    assert response.status_code == 422  # Validation error is correct response


def test_send_message_server_error(headers: dict) -> None:
    """Test if chat send endpoint handles validation errors."""
    url = f"{BASE_URL}/{TEST_TENANT_ID}/api/chat/send"
    data = {
        "user_id": TEST_USER_ID,
        "message": {"content": None, "clicked_option": None},  # Content must be string
        "session_id": "test_session",
    }
    response = requests.post(url, json=data, headers=headers, timeout=(3, 10))
    assert response.status_code == 422  # Validation error for invalid content
