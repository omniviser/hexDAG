"""Robustness of the kernel plugin-port discovery (load_plugin_ports)."""

import importlib

from hexdag.kernel import discovery
from hexdag.kernel.ports.registry import get_port


def test_resolution_and_idempotency():
    discovery.clear_discovery_cache()
    discovery.load_plugin_ports()
    assert get_port("SupportsTransactions") is not None
    # Second call is a cheap no-op (flag set after the scan completes).
    discovery.load_plugin_ports()
    assert get_port("SupportsSessionFactory") is not None


def test_broken_plugin_ports_does_not_break_discovery(monkeypatch):
    """A plugin whose `_ports` raises a non-ImportError must not crash discovery
    for everyone else (Bug 3)."""
    # Populate the registry with a clean load first.
    discovery.clear_discovery_cache()
    discovery.load_plugin_ports()
    assert get_port("SupportsTransactions") is not None

    discovery.clear_discovery_cache()
    real_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name.endswith("._ports"):
            raise RuntimeError("simulated broken plugin _ports")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    try:
        # Must complete without raising despite the broken module import.
        discovery.load_plugin_ports()
        # Previously-registered protocol is still resolvable.
        assert get_port("SupportsTransactions") is not None
    finally:
        # Reset so later code reloads with the real importer.
        discovery.clear_discovery_cache()
