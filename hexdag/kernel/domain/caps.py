"""Capability set model for hexDAG security.

``CapSet`` is a frozen, immutable capability descriptor inspired by Linux
POSIX capabilities.  Capability sets follow a **narrowing-only** principle:
each level (org → pipeline → node → child process) can only restrict
capabilities, never widen them.

Example
-------
.. code-block:: python

    caps = CapSet.unrestricted()
    assert caps.allows("port.llm")

    restricted = caps.revoke("proc.spawn")
    assert not restricted.allows("proc.spawn")

    child = caps.intersect(restricted)  # narrows
    assert not child.allows("proc.spawn")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from hexdag.kernel.exceptions import CapDeniedError

if TYPE_CHECKING:
    from hexdag.kernel.config.models import DefaultCaps

# ============================================================================
# Capability Taxonomy
# ============================================================================

ALL_CAPABILITIES: frozenset[str] = frozenset({
    # VAS operations
    "vas.read",
    "vas.write",
    "vas.exec",
    "vas.list",
    "vas.stat",
    # Port access
    "port.llm",
    "port.tool_router",
    "port.data_store",
    # Process control
    "proc.spawn",
    # Entity operations
    "entity.transition",
    # Memory access
    "mem.read",
    "mem.write",
})

# Human-readable descriptions for each capability
CAPABILITY_DESCRIPTIONS: dict[str, str] = {
    "vas.read": "Read VFS paths",
    "vas.write": "Write to VFS paths",
    "vas.exec": "Execute VFS actions",
    "vas.list": "List VFS directory contents",
    "vas.stat": "Stat VFS path metadata",
    "port.llm": "Access LLM port adapters",
    "port.tool_router": "Access ToolRouter port adapters",
    "port.data_store": "Access DataStore port adapters",
    "proc.spawn": "Spawn child pipelines",
    "entity.transition": "Transition entity states",
    "mem.read": "Read agent memory",
    "mem.write": "Write agent memory",
}


# ============================================================================
# CapSet
# ============================================================================


@dataclass(frozen=True, slots=True)
class CapSet:
    """Immutable capability set for security enforcement.

    A CapSet defines what operations an agent, pipeline, or node is
    permitted to perform.  The ``_denied`` set always takes precedence
    over ``_allowed``.

    Parameters
    ----------
    _allowed : frozenset[str]
        Set of granted capabilities.
    _denied : frozenset[str]
        Set of explicitly denied capabilities (overrides allowed).
    """

    _allowed: frozenset[str] = field(default_factory=frozenset)
    _denied: frozenset[str] = field(default_factory=frozenset)

    # --- Query ---

    def allows(self, cap: str) -> bool:
        """Check whether a capability is granted.

        Parameters
        ----------
        cap : str
            Capability identifier (e.g. ``"port.llm"``).

        Returns
        -------
        bool
            True if *cap* is in the allowed set and NOT in the denied set.
        """
        return cap in self._allowed and cap not in self._denied

    @property
    def allowed(self) -> frozenset[str]:
        """Return the effective allowed capabilities (allowed minus denied)."""
        return self._allowed - self._denied

    @property
    def denied(self) -> frozenset[str]:
        """Return the denied capabilities."""
        return self._denied

    # --- Narrowing ---

    def intersect(self, other: CapSet) -> CapSet:
        """Return a narrowed CapSet — intersection of allowed, union of denied.

        This implements the Linux-style capability dropping principle:
        each level can only restrict, never widen.

        Parameters
        ----------
        other : CapSet
            The CapSet to intersect with.

        Returns
        -------
        CapSet
            New CapSet with narrowed permissions.
        """
        return CapSet(
            _allowed=self._allowed & other._allowed,
            _denied=self._denied | other._denied,
        )

    # --- Mutation (returns new instances) ---

    def grant(self, cap: str) -> CapSet:
        """Return a new CapSet with *cap* added to the allowed set.

        Parameters
        ----------
        cap : str
            Capability to grant.

        Returns
        -------
        CapSet
            New CapSet with *cap* granted.

        Raises
        ------
        CapDeniedError
            If *cap* is in the denied set.
        """
        if cap in self._denied:
            raise CapDeniedError(cap, self)
        return CapSet(
            _allowed=self._allowed | {cap},
            _denied=self._denied,
        )

    def revoke(self, cap: str) -> CapSet:
        """Return a new CapSet with *cap* removed from the allowed set.

        Parameters
        ----------
        cap : str
            Capability to revoke.

        Returns
        -------
        CapSet
            New CapSet with *cap* revoked.
        """
        return CapSet(
            _allowed=self._allowed - {cap},
            _denied=self._denied,
        )

    # --- Factories ---

    @classmethod
    def unrestricted(cls) -> CapSet:
        """Create an unrestricted CapSet with all capabilities allowed.

        Returns
        -------
        CapSet
            CapSet granting all known capabilities.
        """
        return cls(_allowed=ALL_CAPABILITIES)

    @classmethod
    def from_config(cls, config: DefaultCaps) -> CapSet:
        """Create a CapSet from a ``DefaultCaps`` configuration.

        Parameters
        ----------
        config : DefaultCaps
            Configuration with optional ``default_set`` and ``deny`` lists.

        Returns
        -------
        CapSet
            CapSet reflecting the configuration.
        """
        allowed = frozenset(config.default_set) if config.default_set else ALL_CAPABILITIES
        denied = frozenset(config.deny) if config.deny else frozenset()
        return cls(_allowed=allowed, _denied=denied)

    @classmethod
    def from_profile(cls, profile: dict[str, list[str]]) -> CapSet:
        """Create a CapSet from a named profile definition.

        Parameters
        ----------
        profile : dict[str, list[str]]
            Profile dict with optional ``allow`` and ``deny`` keys.

        Returns
        -------
        CapSet
            CapSet reflecting the profile.

        Examples
        --------
        >>> CapSet.from_profile({"allow": ["vas.read", "vas.list"], "deny": ["vas.exec"]})
        CapSet(...)
        """
        allow_list = profile.get("allow", [])
        deny_list = profile.get("deny", [])

        # If allow contains wildcard "*", grant all
        allowed = ALL_CAPABILITIES if "*" in allow_list else frozenset(allow_list)

        return cls(_allowed=allowed, _denied=frozenset(deny_list))
