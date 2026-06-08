"""Langfuse observability plugin for hexDAG.

Provides ``LangfuseObserver`` — a read-only observer that maps hexDAG events
to Langfuse traces, spans, and generations.  Install with::

    pip install hexdag-plugins[langfuse]

Usage::

    from langfuse import Langfuse
    from hexdag.drivers.observer_manager import LocalObserverManager
    from hexdag_plugins.langfuse import LangfuseObserver

    observer_mgr = LocalObserverManager()
    observer_mgr.register(
        LangfuseObserver(Langfuse(), session_id="user-abc"),
        keep_alive=True,
    )
    system = System.from_yaml("system.yaml", observer_manager=observer_mgr)
"""

from hexdag_plugins.langfuse.observer import LangfuseObserver

__all__ = ["LangfuseObserver"]
