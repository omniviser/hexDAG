#!/usr/bin/env python3
"""Build script for hexdag studio UI.

This script builds the React UI and bundles it into the dist folder
which is served by the FastAPI server.

Usage:
    python -m hexdag.studio.build_ui

Requirements:
    - Node.js 18+
    - npm or pnpm
"""

import subprocess
import sys
from pathlib import Path


def build_ui() -> int:
    """Build the React UI."""
    ui_dir = Path(__file__).parent / "ui"

    if not ui_dir.exists():
        print(f"Error: UI directory not found: {ui_dir}")
        return 1

    print("Building hexdag studio UI...")
    print(f"  Directory: {ui_dir}")

    # Check for package.json
    package_json = ui_dir / "package.json"
    if not package_json.exists():
        print(f"Error: package.json not found: {package_json}")
        return 1

    # Try npm first, then pnpm
    npm_cmd = "npm"
    try:
        subprocess.run([npm_cmd, "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        npm_cmd = "pnpm"
        try:
            subprocess.run([npm_cmd, "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: Neither npm nor pnpm found. Please install Node.js.")
            return 1

    print(f"  Using: {npm_cmd}")

    # Install dependencies
    print("\n[1/2] Installing dependencies...")
    result = subprocess.run(
        [npm_cmd, "install"],
        cwd=ui_dir,
        capture_output=False,
    )
    if result.returncode != 0:
        print("Error: Failed to install dependencies")
        return result.returncode

    # Build
    print("\n[2/2] Building production bundle...")
    result = subprocess.run(
        [npm_cmd, "run", "build"],
        cwd=ui_dir,
        capture_output=False,
    )
    if result.returncode != 0:
        print("Error: Failed to build")
        return result.returncode

    dist_dir = ui_dir / "dist"
    if dist_dir.exists():
        print("\nBuild complete!")
        print(f"  Output: {dist_dir}")

        # List built files
        print("\nBuilt files:")
        for f in dist_dir.rglob("*"):
            if f.is_file():
                size = f.stat().st_size
                rel_path = f.relative_to(dist_dir)
                print(f"  {rel_path} ({size:,} bytes)")
    else:
        print("Warning: dist directory not created")

    return 0


if __name__ == "__main__":
    sys.exit(build_ui())
