"""Function to parsing a pyproject.toml file for extra dependencies"""

import re
import tomllib
from pathlib import Path


def find_pyproject(start: Path | None = None) -> Path:
    p = (start or Path.cwd()).resolve()
    for parent in [p, *p.parents]:
        candidate = parent / "pyproject.toml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"pyproject.toml not found starting from {p}")


def get_feature_map(pyproject_path: str | Path | None = None) -> dict[str, str]:
    path = Path(pyproject_path) if pyproject_path else find_pyproject()
    with path.open("rb") as f:
        pyproject_data = tomllib.load(f)

    extras = pyproject_data.get("project", {}).get("optional-dependencies", {})
    feature_map: dict[str, str] = {}
    for feature_name, deps in extras.items():
        for dep in deps:
            raw = dep.strip()
            base = raw.split("[", 1)[0]
            pkg_name = re.split(r"[<>=!~\s]", base, 1)[0]
            if pkg_name == "pyyaml":
                pkg_name = "yyaml"
            feature_map[pkg_name] = feature_name
    return feature_map


FEATURES = get_feature_map()
print(FEATURES)
