"""Tests for SystemBuilder — kind: System YAML compilation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from hexdag.compiler.system_builder import SystemBuilder, SystemBuildError

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def tmp_system(tmp_path: Path) -> Path:
    """Create minimal pipeline YAML files for testing."""
    (tmp_path / "extract.yaml").write_text(
        "apiVersion: hexdag/v1\nkind: Pipeline\nmetadata:\n  name: extract\nspec:\n  nodes: []\n"
    )
    (tmp_path / "transform.yaml").write_text(
        "apiVersion: hexdag/v1\nkind: Pipeline\nmetadata:\n  name: transform\nspec:\n  nodes: []\n"
    )
    (tmp_path / "load.yaml").write_text(
        "apiVersion: hexdag/v1\nkind: Pipeline\nmetadata:\n  name: load\nspec:\n  nodes: []\n"
    )
    return tmp_path


VALID_SYSTEM_YAML = """\
apiVersion: hexdag/v1
kind: System
metadata:
  name: etl-system
spec:
  processes:
    - name: extract
      pipeline: extract.yaml
    - name: transform
      pipeline: transform.yaml
    - name: load
      pipeline: load.yaml
  pipes:
    - from: extract
      to: transform
      mapping:
        records: "{{ extract.records }}"
    - from: transform
      to: load
      mapping:
        results: "{{ transform.results }}"
"""


class TestSystemBuilderParsing:
    def test_build_from_string(self, tmp_system: Path) -> None:
        builder = SystemBuilder(base_path=tmp_system)
        config = builder.build_from_yaml_string(VALID_SYSTEM_YAML)
        assert config.metadata["name"] == "etl-system"
        assert len(config.processes) == 3
        assert len(config.pipes) == 2

    def test_build_from_file(self, tmp_system: Path) -> None:
        yaml_file = tmp_system / "system.yaml"
        yaml_file.write_text(VALID_SYSTEM_YAML)
        builder = SystemBuilder()
        config = builder.build_from_yaml_file(yaml_file)
        assert config.metadata["name"] == "etl-system"

    def test_rejects_wrong_kind(self, tmp_system: Path) -> None:
        builder = SystemBuilder(base_path=tmp_system)
        with pytest.raises(SystemBuildError, match="Expected kind: System"):
            builder.build_from_yaml_string(
                "kind: Pipeline\nmetadata:\n  name: x\nspec:\n  nodes: []\n"
            )

    def test_rejects_missing_spec(self, tmp_system: Path) -> None:
        builder = SystemBuilder(base_path=tmp_system)
        with pytest.raises(SystemBuildError, match="missing 'spec'"):
            builder.build_from_yaml_string("kind: System\nmetadata:\n  name: x\n")

    def test_rejects_missing_file(self) -> None:
        builder = SystemBuilder()
        with pytest.raises(SystemBuildError, match="not found"):
            builder.build_from_yaml_file("/nonexistent/system.yaml")

    def test_rejects_missing_pipeline_path(self, tmp_system: Path) -> None:
        yaml_content = """\
kind: System
metadata:
  name: test
spec:
  processes:
    - name: missing
      pipeline: nonexistent.yaml
"""
        builder = SystemBuilder(base_path=tmp_system)
        with pytest.raises(SystemBuildError, match="missing pipeline"):
            builder.build_from_yaml_string(yaml_content)


class TestSystemBuilderCycleDetection:
    def test_rejects_cycle(self, tmp_system: Path) -> None:
        yaml_content = """\
kind: System
metadata:
  name: cyclic
spec:
  processes:
    - name: extract
      pipeline: extract.yaml
    - name: transform
      pipeline: transform.yaml
  pipes:
    - from: extract
      to: transform
      mapping: {}
    - from: transform
      to: extract
      mapping: {}
"""
        builder = SystemBuilder(base_path=tmp_system)
        with pytest.raises(SystemBuildError, match="Cycle detected"):
            builder.build_from_yaml_string(yaml_content)

    def test_accepts_diamond(self, tmp_system: Path) -> None:
        """Diamond DAG (a -> b, a -> c, b -> d, c -> d) is valid."""
        for name in ("a", "b", "c", "d"):
            (tmp_system / f"{name}.yaml").write_text(
                f"apiVersion: hexdag/v1\nkind: Pipeline\nmetadata:\n  name: {name}\nspec:\n"
                "nodes: []\n"
            )
        yaml_content = """\
kind: System
metadata:
  name: diamond
spec:
  processes:
    - name: a
      pipeline: a.yaml
    - name: b
      pipeline: b.yaml
    - name: c
      pipeline: c.yaml
    - name: d
      pipeline: d.yaml
  pipes:
    - { from: a, to: b, mapping: {} }
    - { from: a, to: c, mapping: {} }
    - { from: b, to: d, mapping: {} }
    - { from: c, to: d, mapping: {} }
"""
        builder = SystemBuilder(base_path=tmp_system)
        config = builder.build_from_yaml_string(yaml_content)
        assert len(config.processes) == 4


class TestTopologicalOrder:
    def test_linear_chain(self, tmp_system: Path) -> None:
        builder = SystemBuilder(base_path=tmp_system)
        config = builder.build_from_yaml_string(VALID_SYSTEM_YAML)
        order = SystemBuilder.topological_order(config)
        assert order.index("extract") < order.index("transform")
        assert order.index("transform") < order.index("load")

    def test_no_pipes_preserves_declaration_order(self, tmp_system: Path) -> None:
        yaml_content = """\
kind: System
metadata:
  name: parallel
spec:
  processes:
    - name: extract
      pipeline: extract.yaml
    - name: transform
      pipeline: transform.yaml
    - name: load
      pipeline: load.yaml
  pipes: []
"""
        builder = SystemBuilder(base_path=tmp_system)
        config = builder.build_from_yaml_string(yaml_content)
        order = SystemBuilder.topological_order(config)
        assert order == ["extract", "transform", "load"]  # declaration order preserved
