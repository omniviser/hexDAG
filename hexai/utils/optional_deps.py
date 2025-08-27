"""Function to parsing a pyproject.toml file for extra dependencies"""

import re
import tomllib
from pathlib import Path


def get_feature_map(pyproject_path: str | Path = "pyproject.toml") -> dict[str, str]:
    path = Path(pyproject_path)
    with path.open("rb") as f:
        pyproject_data = tomllib.load(f)

    extras = pyproject_data.get("project", {}).get("optional-dependencies", {})
    feature_map: dict[str, str] = {}
    for feature_name, deps in extras.items():
        for dep in deps:
            pkg_name = re.split(r"[<>=~\s]", dep, 1)[0]
            feature_map[pkg_name] = feature_name
    return feature_map


FEATURES = get_feature_map()
