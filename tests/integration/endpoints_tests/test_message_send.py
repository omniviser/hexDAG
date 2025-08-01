"""Test cases for message sending endpoints."""

import pytest
import requests

from ..config import BASE_URL, TEST_TENANT_ID, TEST_USER_ID


def test_send_message_success(headers: dict) -> None:
    """Test successful message sending."""
    url = f"{BASE_URL}/{TEST_TENANT_ID}/api/chat/send"
    data = {
        "user_id": TEST_USER_ID,
        "message": {"content": "Hello, this is a test message", "clicked_option": None},
        "session_id": "test_session",
    }
    response = requests.post(url, json=data, headers=headers, timeout=(3, 10))

    assert response.status_code == 202
    data = response.json()
    assert "task_id" in data
    assert "session_id" in data


def test_send_message_accepted(headers: dict) -> None:
    """Test message sending with accepted status (202)."""
    url = f"{BASE_URL}/{TEST_TENANT_ID}/api/chat/send"
    data = {
        "user_id": TEST_USER_ID,
        "message": {
            "content": "This is a long message that will be processed asynchronously",
            "clicked_option": None,
        },
        "session_id": "test_session",
    }
    response = requests.post(url, json=data, headers=headers, timeout=(3, 10))
    assert response.status_code == 202
    data = response.json()
    assert "task_id" in data
    assert "session_id" in data


def test_send_message_empty_content(headers: dict) -> None:
    """Test message sending with empty content."""
    url = f"{BASE_URL}/{TEST_TENANT_ID}/api/chat/send"
    data = {
        "user_id": TEST_USER_ID,
        "message": {"content": "", "clicked_option": None},
        "session_id": "test_session",
    }
    response = requests.post(url, json=data, headers=headers, timeout=(3, 10))
    assert response.status_code == 422  # Empty content is invalid


def test_send_message_missing_session_id(headers: dict) -> None:
    """Test message sending without session ID."""
    url = f"{BASE_URL}/{TEST_TENANT_ID}/api/chat/send"
    data = {"user_id": TEST_USER_ID, "message": {"content": "Test message", "clicked_option": None}}
    response = requests.post(url, json=data, headers=headers, timeout=(3, 10))
    assert response.status_code == 202  # Session ID is optional, will be generated
    data = response.json()
    assert "session_id" in data  # Should return a generated session ID


@pytest.mark.skip(reason="Auth flow being updated")
def test_send_message_unauthorized() -> None:
    """Test message sending without authorization."""
    url = f"{BASE_URL}/{TEST_TENANT_ID}/api/chat/send"
    data = {
        "user_id": TEST_USER_ID,
        "message": {"content": "Test message", "clicked_option": None},
        "session_id": "test_session",
    }
    response = requests.post(url, json=data, timeout=(3, 10))
    assert response.status_code == 401


@pytest.mark.skip(reason="Auth flow being updated")
def test_send_message_forbidden(headers: dict) -> None:
    """Test message sending with invalid permissions."""
    url = f"{BASE_URL}/{TEST_TENANT_ID}/api/chat/send"
    data = {
        "user_id": TEST_USER_ID,
        "message": {"content": "Test message", "clicked_option": None},
        "session_id": "test_session",
    }
    response = requests.post(url, json=data, headers=headers, timeout=(3, 10))
    assert response.status_code == 403


def test_send_message_validation_error(headers: dict) -> None:
    """Test message sending with invalid parameters."""
    url = f"{BASE_URL}/{TEST_TENANT_ID}/api/chat/send"
    data = {
        "user_id": TEST_USER_ID,
        "message": {"content": None, "clicked_option": None},  # Content must be string
        "session_id": "test_session",
    }
    response = requests.post(url, json=data, headers=headers, timeout=(3, 10))
    assert response.status_code == 422
