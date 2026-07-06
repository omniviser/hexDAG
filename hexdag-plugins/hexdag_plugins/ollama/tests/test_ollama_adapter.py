"""Shim tests: plugin path re-exports the stdlib Ollama adapter.

Full adapter tests live in ``tests/hexdag/stdlib/adapters/ollama/`` in the
main hexdag repo.
"""

import importlib
import sys
import warnings


def _fresh_import(module_name: str):
    for mod in list(sys.modules):
        if mod.startswith("hexdag_plugins.ollama"):
            del sys.modules[mod]
    return importlib.import_module(module_name)


def test_plugin_import_warns_and_aliases_stdlib():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = _fresh_import("hexdag_plugins.ollama.adapters.ollama")

    from hexdag.stdlib.adapters.ollama.ollama_adapter import OllamaAdapter

    assert module.OllamaAdapter is OllamaAdapter
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_package_reexport_still_works():
    module = _fresh_import("hexdag_plugins.ollama")

    from hexdag.stdlib.adapters.ollama import OllamaAdapter

    assert module.OllamaAdapter is OllamaAdapter
