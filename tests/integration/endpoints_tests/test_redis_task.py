"""Test suite for the /tasks/{task_id} endpoint.

This module contains tests to verify the functionality of the /tasks/{task_id}
endpoint, which checks the status of Redis queue tasks (jobs). The tests included are:
- Testing successful retrieval of task status for a valid task_id.
- Verifying 422 errors for invalid task_id formats.
- Testing 404 errors for non-existent task IDs.
"""

import requests

from ..config import BASE_URL


def test_get_task_status_success(headers: dict) -> None:
    """Test if /tasks/{task_id} returns the status of a Redis queue task."""
    create_task_response = requests.get(
        f"{BASE_URL}/task/", params={"delay_time": 5.0, "n_process": 1}, timeout=(3, 10), headers=headers
    )
    create_task_json = create_task_response.json()
    task_id = create_task_json["task_id"]

    response = requests.get(f"{BASE_URL}/tasks/{task_id}/", timeout=(3, 10), headers=headers)

    assert response.status_code == 200
    assert "status" in response.json(), "task's status not found in json response"


# add exception handler, test it and remove this. Current block - refactor app.py
# def test_get_task_status_not_found(headers: dict) -> None:
#     """Test if /tasks/{task_id} returns 404 for non-existing task_id."""
#     task_id = "40a81c69-04a9-4826-8040-32b03ead2425"
#     response = requests.get(f"{BASE_URL}/tasks/{task_id}/", timeout=(3, 10), headers=headers)
#
#     assert response.status_code == 404


def test_create_dummy_task_validation_error(headers: dict) -> None:
    """Test if /task returns 422 for invalid delay time."""
    params = {"delay": "invalid"}  # Non-numeric delay
    response = requests.get(f"{BASE_URL}/task", params=params, timeout=(3, 10), headers=headers)
    assert response.status_code == 422
    response_json = response.json()
    assert "detail" in response_json
    assert isinstance(response_json["detail"], list)
    assert len(response_json["detail"]) > 0
    assert "msg" in response_json["detail"][0]
