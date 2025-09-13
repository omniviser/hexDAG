# Release HOWTO 🚀

## Local
- pre-commit: `uv run pre-commit run --all-files`
- tests: `uv run pytest`
- build: `uv build`
- twine: `uvx twine check dist/*`

## CI
- Matrix build + smoke install (artifacts: `dist-<OS>`).
- Deptry report (artifact).
- CHANGELOG, PACKAGING_PLAN (artifacts).

## Release
- Tag `vX.Y.Z` → pipeline publishes to TestPyPI.
- Prod PyPI: disabled job `PublishProdPyPI` (to be enabled later).
- Import time <100 ms: **DEFERRED** until codebase stabilization.
