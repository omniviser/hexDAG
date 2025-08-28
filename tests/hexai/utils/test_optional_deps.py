import textwrap
from pathlib import Path

from hexai.utils.optional_deps import get_feature_to_pkg, get_pkg_feature


def _write_pyproject(tmp_path: Path) -> Path:
    # Small pyproject covering version operators and empty extras
    content = textwrap.dedent(
        """
        [project]
        name = "hexdag"

        [project.optional-dependencies]
        cli = ["pyyaml~=6.0", "rich"]
        viz = ["graphviz>=0.20"]
        empty = []
    """
    ).strip()
    p = tmp_path / "pyproject.toml"
    p.write_text(content, encoding="utf-8")
    return p


def test_get_feature_to_pkg(tmp_path: Path):
    mapping = get_feature_to_pkg(_write_pyproject(tmp_path))
    assert mapping == {
        "cli": ["pyyaml", "rich"],
        "viz": ["graphviz"],
        "empty": [],
    }


def test_get_pkg_feature(tmp_path: Path):
    mapping = get_pkg_feature(_write_pyproject(tmp_path))
    assert mapping == {
        "pyyaml": "cli",
        "rich": "cli",
        "graphviz": "viz",
    }
