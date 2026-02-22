#!/usr/bin/env python3
"""Detect which monorepo packages have changes since their last release tag.

Checks git history to find the latest release tag for each package, then
reports which packages have file changes in their source directories.

Outputs a JSON object with changed package names as keys and their tag info
as values, suitable for use in GitHub Actions matrix strategies.
"""

import json
import subprocess
import sys

PACKAGES: dict[str, dict[str, str]] = {
    "hexdag": {
        "tag_prefix": "v",
        "paths": "hexdag/",
    },
    "hexdag-plugins": {
        "tag_prefix": "plugins-v",
        "paths": "hexdag_plugins/",
    },
    "hexdag-studio": {
        "tag_prefix": "studio-v",
        "paths": "hexdag-studio/",
    },
}


def run(cmd: str) -> str:
    """Run a shell command and return stripped stdout."""
    result = subprocess.run(  # noqa: S603
        cmd.split(),
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def get_latest_tag(prefix: str) -> str | None:
    """Find the most recent git tag matching a prefix."""
    tags = run(f"git tag -l --sort=-v:refname {prefix}*")
    if not tags:
        return None
    return tags.split("\n")[0]


def has_changes(tag: str | None, paths: str) -> bool:
    """Check if there are changes in paths since tag (or ever, if no tag)."""
    if tag is None:
        # No previous release — everything is a change
        return True
    diff = run(f"git diff --name-only {tag}..HEAD -- {paths}")
    return bool(diff)


def main() -> None:
    changed: list[str] = []

    for name, config in PACKAGES.items():
        tag = get_latest_tag(config["tag_prefix"])
        tag_display = tag or "(no previous release)"

        if has_changes(tag, config["paths"]):
            changed.append(name)
            print(  # noqa: T201
                f"  {name}: CHANGED since {tag_display}",
                file=sys.stderr,
            )
        else:
            print(  # noqa: T201
                f"  {name}: no changes since {tag_display}",
                file=sys.stderr,
            )

    # Output for GitHub Actions
    print(json.dumps(changed))  # noqa: T201


if __name__ == "__main__":
    sys.exit(main() or 0)
