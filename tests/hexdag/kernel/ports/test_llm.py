"""Tests for Literal type constraints in LLM port models."""

import pytest
from pydantic import ValidationError

from hexdag.kernel.ports.llm import ImageContent, Message, VisionMessage


class TestMessageRole:
    @pytest.mark.parametrize("role", ["user", "assistant", "system", "tool", "human", "ai"])
    def test_valid_roles(self, role: str) -> None:
        msg = Message(role=role, content="test")
        assert msg.role == role

    def test_invalid_role_raises(self) -> None:
        with pytest.raises(ValidationError):
            Message(role="invalid_role", content="test")


class TestVisionMessageRole:
    @pytest.mark.parametrize("role", ["user", "assistant", "system", "tool", "human", "ai"])
    def test_valid_roles(self, role: str) -> None:
        msg = VisionMessage(role=role, content="test")
        assert msg.role == role

    def test_invalid_role_raises(self) -> None:
        with pytest.raises(ValidationError):
            VisionMessage(role="invalid_role", content="test")


class TestImageContentType:
    @pytest.mark.parametrize("image_type", ["image", "image_url"])
    def test_valid_types(self, image_type: str) -> None:
        ic = ImageContent(type=image_type, source="test.jpg")
        assert ic.type == image_type

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            ImageContent(type="video", source="test.mp4")

    def test_default_type(self) -> None:
        ic = ImageContent(source="test.jpg")
        assert ic.type == "image"


class TestImageDetail:
    @pytest.mark.parametrize("detail", ["low", "high", "auto"])
    def test_valid_details(self, detail: str) -> None:
        ic = ImageContent(source="test.jpg", detail=detail)
        assert ic.detail == detail

    def test_invalid_detail_raises(self) -> None:
        with pytest.raises(ValidationError):
            ImageContent(source="test.jpg", detail="ultra")

    def test_default_detail(self) -> None:
        ic = ImageContent(source="test.jpg")
        assert ic.detail == "auto"
