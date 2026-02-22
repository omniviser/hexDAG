"""hexdag-studio: Visual Studio UI for hexDAG pipelines."""

try:
    from importlib.metadata import version as _meta_version

    __version__ = _meta_version("hexdag-studio")
except Exception:
    __version__ = "0.0.0.dev0"  # Fallback for development installs
