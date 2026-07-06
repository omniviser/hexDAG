"""Shim tests: plugin path re-exports the stdlib Vertex adapter.

Full adapter tests live in ``tests/hexdag/stdlib/adapters/google/`` in the
main hexdag repo.
"""

import importlib
import sys
import warnings


def _fresh_import(module_name: str):
    for mod in list(sys.modules):
        if mod.startswith("hexdag_plugins.google"):
            del sys.modules[mod]
    return importlib.import_module(module_name)


def test_plugin_import_warns_and_aliases_stdlib():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = _fresh_import("hexdag_plugins.google.adapters.vertex")

    from hexdag.stdlib.adapters.google.vertex_adapter import (
        VertexAIAdapter,
        aclose_all_vertex_clients,
    )

    assert module.VertexAIAdapter is VertexAIAdapter
    assert module.aclose_all_vertex_clients is aclose_all_vertex_clients
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_package_reexport_still_works():
    module = _fresh_import("hexdag_plugins.google")

    from hexdag.stdlib.adapters.google import VertexAIAdapter

    assert module.VertexAIAdapter is VertexAIAdapter
