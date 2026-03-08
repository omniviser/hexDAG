"""Tests for CheckpointNode -- declarative mid-pipeline save/restore."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hexdag.kernel.domain.dag import NodeSpec
from hexdag.stdlib.nodes.checkpoint_node import CheckpointNode

# ---------------------------------------------------------------------------
# TestCheckpointNodeCreation
# ---------------------------------------------------------------------------


class TestCheckpointNodeCreation:
    """Tests for CheckpointNode factory __call__."""

    def setup_method(self) -> None:
        self.factory = CheckpointNode()

    def test_creates_save_node(self) -> None:
        """Creates a valid NodeSpec for save action."""
        spec = self.factory(
            name="save_ckpt",
            action="save",
            run_id="run_123",
            keys=["extract_data"],
        )
        assert isinstance(spec, NodeSpec)
        assert spec.name == "save_ckpt"
        assert spec.fn is not None

    def test_creates_restore_node(self) -> None:
        """Creates a valid NodeSpec for restore action."""
        spec = self.factory(
            name="restore_ckpt",
            action="restore",
            run_id="run_123",
        )
        assert isinstance(spec, NodeSpec)
        assert spec.name == "restore_ckpt"

    def test_includes_dependencies(self) -> None:
        """Dependencies are passed through to NodeSpec."""
        spec = self.factory(
            name="save",
            action="save",
            run_id="run_1",
            deps=["transform_data"],
        )
        assert "transform_data" in spec.deps

    def test_empty_deps_default(self) -> None:
        """No deps defaults to empty frozenset."""
        spec = self.factory(
            name="save",
            action="save",
            run_id="run_1",
        )
        assert spec.deps == frozenset()


# ---------------------------------------------------------------------------
# TestCheckpointSave (async)
# ---------------------------------------------------------------------------


class TestCheckpointSave:
    """Tests for checkpoint save execution."""

    def setup_method(self) -> None:
        self.factory = CheckpointNode()

    @pytest.mark.asyncio
    async def test_save_skipped_without_port(self) -> None:
        """Save is skipped when no checkpoint port is configured."""
        spec = self.factory(
            name="save_ckpt",
            action="save",
            run_id="run_1",
        )
        with patch("hexdag.stdlib.nodes.checkpoint_node.get_user_ports", return_value={}):
            result = await spec.fn({"key": "value"})
        assert result["skipped"] is True

    @pytest.mark.asyncio
    async def test_save_persists_selected_keys(self) -> None:
        """Save persists only specified keys from input_data."""
        mock_storage = MagicMock()
        mock_storage.aset = AsyncMock()
        mock_storage.aget = AsyncMock(return_value=None)

        spec = self.factory(
            name="save_ckpt",
            action="save",
            run_id="run_1",
            keys=["extract_data"],
        )

        with (
            patch(
                "hexdag.stdlib.nodes.checkpoint_node.get_user_ports",
                return_value={"checkpoint": mock_storage},
            ),
            patch(
                "hexdag.kernel.orchestration.components.checkpoint_manager.CheckpointManager.save",
                new_callable=AsyncMock,
            ),
        ):
            result = await spec.fn(
                {"extract_data": {"items": [1, 2]}, "other_data": "ignored"},
                checkpoint=mock_storage,
            )

        assert result["saved"] is True
        assert result["run_id"] == "run_1"
        assert "extract_data" in result["keys"]

    @pytest.mark.asyncio
    async def test_save_all_keys_when_none(self) -> None:
        """Save all keys from input_data when keys=None."""
        mock_storage = MagicMock()

        spec = self.factory(
            name="save_all",
            action="save",
            run_id="run_1",
            keys=None,
        )

        with (
            patch(
                "hexdag.stdlib.nodes.checkpoint_node.get_user_ports",
                return_value={"checkpoint": mock_storage},
            ),
            patch(
                "hexdag.kernel.orchestration.components.checkpoint_manager.CheckpointManager.save",
                new_callable=AsyncMock,
            ),
        ):
            result = await spec.fn(
                {"a": 1, "b": 2, "c": 3},
                checkpoint=mock_storage,
            )

        assert result["saved"] is True
        assert set(result["keys"]) == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# TestCheckpointRestore (async)
# ---------------------------------------------------------------------------


class TestCheckpointRestore:
    """Tests for checkpoint restore execution."""

    def setup_method(self) -> None:
        self.factory = CheckpointNode()

    @pytest.mark.asyncio
    async def test_restore_skipped_without_port(self) -> None:
        """Restore is skipped when no checkpoint port is configured."""
        spec = self.factory(
            name="restore_ckpt",
            action="restore",
            run_id="run_1",
        )
        with patch("hexdag.stdlib.nodes.checkpoint_node.get_user_ports", return_value={}):
            result = await spec.fn({})
        assert result["skipped"] is True

    @pytest.mark.asyncio
    async def test_restore_not_found(self) -> None:
        """Restore returns found=False when checkpoint doesn't exist."""
        mock_storage = MagicMock()

        spec = self.factory(
            name="restore_ckpt",
            action="restore",
            run_id="nonexistent_run",
        )

        with (
            patch(
                "hexdag.stdlib.nodes.checkpoint_node.get_user_ports",
                return_value={"checkpoint": mock_storage},
            ),
            patch(
                "hexdag.kernel.orchestration.components.checkpoint_manager.CheckpointManager.load",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await spec.fn({}, checkpoint=mock_storage)

        assert result["found"] is False
        assert result["run_id"] == "nonexistent_run"


# ---------------------------------------------------------------------------
# TestRunIdTemplateResolution
# ---------------------------------------------------------------------------


class TestRunIdTemplateResolution:
    """Tests for run_id template resolution."""

    def setup_method(self) -> None:
        self.factory = CheckpointNode()

    @pytest.mark.asyncio
    async def test_resolves_template_in_run_id(self) -> None:
        """Template {{key}} in run_id is resolved from input_data."""
        mock_storage = MagicMock()

        spec = self.factory(
            name="restore_ckpt",
            action="restore",
            run_id="{{ metadata.run_id }}",
        )

        with (
            patch(
                "hexdag.stdlib.nodes.checkpoint_node.get_user_ports",
                return_value={"checkpoint": mock_storage},
            ),
            patch(
                "hexdag.kernel.orchestration.components.checkpoint_manager.CheckpointManager.load",
                new_callable=AsyncMock,
                return_value=None,
            ) as mock_load,
        ):
            await spec.fn(
                {"metadata": {"run_id": "resolved_123"}},
                checkpoint=mock_storage,
            )

        # load should be called with the resolved run_id
        mock_load.assert_called_once_with("resolved_123")

    @pytest.mark.asyncio
    async def test_static_run_id_unchanged(self) -> None:
        """Static run_id without templates passes through unchanged."""
        mock_storage = MagicMock()

        spec = self.factory(
            name="restore_ckpt",
            action="restore",
            run_id="static_run_42",
        )

        with (
            patch(
                "hexdag.stdlib.nodes.checkpoint_node.get_user_ports",
                return_value={"checkpoint": mock_storage},
            ),
            patch(
                "hexdag.kernel.orchestration.components.checkpoint_manager.CheckpointManager.load",
                new_callable=AsyncMock,
                return_value=None,
            ) as mock_load,
        ):
            await spec.fn({}, checkpoint=mock_storage)

        mock_load.assert_called_once_with("static_run_42")
