"""Shared fixtures for hexdag_studio tests."""

import sys
from pathlib import Path
from typing import Generator

import pytest

# Add hexdag-studio to path for import to work with pytest importlib mode
# This is needed because hexdag-studio is a separate package with its own pyproject.toml
_studio_root = Path(__file__).parent.parent.parent / "hexdag-studio"
if str(_studio_root) not in sys.path:
    sys.path.insert(0, str(_studio_root))

# Force reimport if the module was cached without the path
if "hexdag_studio" in sys.modules:
    del sys.modules["hexdag_studio"]
if "hexdag_studio.server" in sys.modules:
    del sys.modules["hexdag_studio.server"]
if "hexdag_studio.server.main" in sys.modules:
    del sys.modules["hexdag_studio.server.main"]

from fastapi.testclient import TestClient  # noqa: E402

from hexdag_studio.server.main import create_app  # noqa: E402
from hexdag_studio.server.routes.files import _WorkspaceConfig  # noqa: E402


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary workspace directory with a sample pipeline."""
    # Create workspace structure
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Create a sample pipeline file
    sample_pipeline = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-pipeline
spec:
  nodes:
    - kind: data_node
      metadata:
        name: start
      spec:
        output:
          value: "hello"
      dependencies: []
"""
    (workspace / "test_pipeline.yaml").write_text(sample_pipeline)

    # Create a subdirectory with another pipeline
    subdir = workspace / "subdir"
    subdir.mkdir()
    (subdir / "nested.yaml").write_text(sample_pipeline)

    yield workspace

    # Cleanup is automatic with tmp_path


@pytest.fixture
def client(temp_workspace: Path) -> Generator[TestClient, None, None]:
    """Create a test client with the temporary workspace."""
    app = create_app(workspace_path=temp_workspace)
    with TestClient(app) as client:
        yield client
    # Reset workspace config after test
    _WorkspaceConfig.reset()


@pytest.fixture
def client_no_workspace() -> Generator[TestClient, None, None]:
    """Create a test client without workspace configured."""
    # Reset any existing config
    _WorkspaceConfig.reset()

    # Create app with a non-existent workspace (will fail on workspace operations)
    app = create_app(workspace_path=Path("/nonexistent"))
    with TestClient(app) as client:
        yield client
    _WorkspaceConfig.reset()
