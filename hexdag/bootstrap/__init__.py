"""Self-compiling bootstrap for hexDAG.

hexDAG is a self-compiling operating system: it uses its own pipeline
orchestration to build, validate, test, and package itself.  This is the
classic bootstrapping pattern -- like GCC compiling GCC or the Linux kernel
built by ``make`` running on Linux.

The bootstrap sequence:

1. **Stage 0** -- A minimal Python runtime imports hexDAG's kernel.
2. **Stage 1** -- The kernel (orchestrator, pipeline builder) loads.
3. **Stage 2** -- hexDAG reads ``self_compile.yaml`` and builds a
   :class:`~hexdag.kernel.domain.dag.DirectedGraph` from it.
4. **Stage 3** -- The orchestrator executes the self-compile DAG, which
   lints, type-checks, validates architecture, runs tests, and builds
   the package -- all as deterministic DAG nodes.

The meta-recursive proof: one of the stages (``validate_self``) validates
the bootstrap pipeline *itself* using hexDAG's own pipeline validator,
proving the system is self-consistent.

Quick start::

    # From CLI
    hexdag bootstrap

    # Programmatic
    from hexdag.bootstrap.runner import run_self_compile
    results = await run_self_compile()
"""

from hexdag.bootstrap.runner import run_self_compile
from hexdag.bootstrap.service import BootstrapService

__all__ = ["BootstrapService", "run_self_compile"]
