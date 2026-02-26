"""Tests for SystemConfig, ProcessSpec, PipeSpec, and Pipe domain models."""

from __future__ import annotations

import pytest

from hexdag.kernel.domain.system_config import (
    Pipe,
    PipeSpec,
    ProcessSpec,
    SystemConfig,
)


class TestPipe:
    def test_frozen_dataclass(self) -> None:
        pipe = Pipe(from_process="a", to_process="b", mapping={"x": "{{ a.x }}"})
        assert pipe.from_process == "a"
        assert pipe.to_process == "b"
        assert pipe.mapping == {"x": "{{ a.x }}"}

    def test_immutable(self) -> None:
        pipe = Pipe(from_process="a", to_process="b")
        with pytest.raises(AttributeError):
            pipe.from_process = "c"  # type: ignore[misc]

    def test_default_mapping(self) -> None:
        pipe = Pipe(from_process="a", to_process="b")
        assert pipe.mapping == {}


class TestPipeSpec:
    def test_parse_from_yaml_style_dict(self) -> None:
        spec = PipeSpec.model_validate({
            "from": "extract",
            "to": "transform",
            "mapping": {"records": "{{ extract.records }}"},
        })
        assert spec.from_process == "extract"
        assert spec.to_process == "transform"

    def test_to_domain(self) -> None:
        spec = PipeSpec.model_validate({
            "from": "a",
            "to": "b",
            "mapping": {"x": "{{ a.x }}"},
        })
        pipe = spec.to_domain()
        assert isinstance(pipe, Pipe)
        assert pipe.from_process == "a"
        assert pipe.to_process == "b"
        assert pipe.mapping == {"x": "{{ a.x }}"}


class TestProcessSpec:
    def test_minimal(self) -> None:
        spec = ProcessSpec(name="extract", pipeline="./extract.yaml")
        assert spec.name == "extract"
        assert spec.ports == {}
        assert spec.input_schema is None

    def test_full(self) -> None:
        spec = ProcessSpec(
            name="transform",
            pipeline="./transform.yaml",
            input_schema={"records": "array"},
            output_schema={"results": "array"},
            ports={"llm": {"namespace": "core", "name": "openai"}},
        )
        assert spec.input_schema == {"records": "array"}
        assert "llm" in spec.ports


class TestSystemConfig:
    def _make_config(self, **overrides: object) -> SystemConfig:
        defaults: dict[str, object] = {
            "metadata": {"name": "test-system"},
            "processes": [
                {"name": "a", "pipeline": "./a.yaml"},
                {"name": "b", "pipeline": "./b.yaml"},
            ],
            "pipes": [{"from": "a", "to": "b", "mapping": {"x": "{{ a.x }}"}}],
        }
        defaults.update(overrides)
        return SystemConfig.model_validate(defaults)

    def test_valid_config(self) -> None:
        config = self._make_config()
        assert len(config.processes) == 2
        assert len(config.pipes) == 1

    def test_process_names(self) -> None:
        config = self._make_config()
        assert config.process_names == ["a", "b"]

    def test_domain_pipes(self) -> None:
        config = self._make_config()
        pipes = config.domain_pipes
        assert len(pipes) == 1
        assert isinstance(pipes[0], Pipe)
        assert pipes[0].from_process == "a"

    def test_rejects_unknown_from_process(self) -> None:
        with pytest.raises(ValueError, match="unknown process 'missing'"):
            self._make_config(
                pipes=[{"from": "missing", "to": "b", "mapping": {}}],
            )

    def test_rejects_unknown_to_process(self) -> None:
        with pytest.raises(ValueError, match="unknown process 'missing'"):
            self._make_config(
                pipes=[{"from": "a", "to": "missing", "mapping": {}}],
            )

    def test_rejects_self_loop(self) -> None:
        with pytest.raises(ValueError, match="cannot connect process 'a' to itself"):
            self._make_config(
                pipes=[{"from": "a", "to": "a", "mapping": {}}],
            )

    def test_rejects_duplicate_process_names(self) -> None:
        with pytest.raises(ValueError, match="Duplicate process name"):
            self._make_config(
                processes=[
                    {"name": "a", "pipeline": "./a.yaml"},
                    {"name": "a", "pipeline": "./a2.yaml"},
                ],
                pipes=[],
            )

    def test_no_pipes_is_valid(self) -> None:
        config = self._make_config(pipes=[])
        assert len(config.pipes) == 0

    def test_global_ports(self) -> None:
        config = self._make_config(
            ports={"llm": {"namespace": "core", "name": "openai"}},
        )
        assert "llm" in config.ports
