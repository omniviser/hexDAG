"""Function to parsing a pyproject.toml file for extra dependencies."""

import re
import tomllib
from pathlib import Path


def find_pyproject(start: Path | None = None) -> Path:
    """Find a pyproject.toml file."""
    p = (start or Path.cwd()).resolve()
    for parent in [p, *p.parents]:
        candidate = parent / "pyproject.toml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"pyproject.toml not found starting from {p}")


def load_pyproject(pyproject_path: str | Path | None = None) -> dict:
    """Load a pyproject.toml file."""
    path = Path(pyproject_path) if pyproject_path else find_pyproject()
    with path.open("rb") as f:
        pyproject_data = tomllib.load(f)

    return pyproject_data.get("project", {}).get("optional-dependencies", {})


def dep_clear(dep: str) -> str:
    """Clear to raw dependencies."""
    raw = dep.strip()
    base = raw.split("[", maxsplit=1)[0]
    return re.split(r"[<>=!~\s]", base, maxsplit=1)[0]


def get_pkg_feature(pyproject_path: str | Path | None = None) -> dict[str, str]:
    """Map of clean package names to the feature that declares them.

    Example
    -------
    {"pyyaml": "cli", "graphviz": "viz"}.
    """
    extras = load_pyproject(pyproject_path)
    feature_map: dict[str, str] = {}
    for feature_name, deps in extras.items():
        for dep in deps:
            pkg_name = dep_clear(dep)
            feature_map[pkg_name] = feature_name
    return feature_map


def get_feature_to_pkg(pyproject_path: str | Path | None = None) -> dict[str, list[str]]:
    """Map feature -> list of clean package names.

    Example
    -------
    {"cli": ["pyyaml"], "viz": ["graphviz"], "adapters-openai": []}.
    """
    extras = load_pyproject(pyproject_path)

    return {feature_name: [dep_clear(d) for d in deps] for feature_name, deps in extras.items()}
