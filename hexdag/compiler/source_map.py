"""Source provenance for compiled YAML documents.

Maps structural paths (tuples like ``("spec", "nodes", 3, "kind")``) to
:class:`~hexdag.compiler.diagnostics.Location` objects recorded during
parsing. Because ``!include`` expansion happens at parse time, locations
point into the *original* file an element came from — including fragments.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hexdag.compiler.diagnostics import Location

type SourcePath = tuple[str | int, ...]


@dataclass(slots=True)
class SourceMap:
    """Path-keyed source locations with longest-prefix fallback lookup."""

    _locations: dict[SourcePath, Location] = field(default_factory=dict)

    def record(self, path: SourcePath, loc: Location) -> None:
        """Record the location of the value at *path*."""
        self._locations[path] = loc

    def at(self, path: SourcePath) -> Location | None:
        """Location for *path*, falling back to the nearest recorded ancestor.

        Rules may hand in approximate paths; walking up the tuple still
        yields a nearby line, which beats no location at all.
        """
        probe = tuple(path)
        while True:
            if probe in self._locations:
                loc = self._locations[probe]
                return Location(file=loc.file, line=loc.line, column=loc.column, path=tuple(path))
            if not probe:
                return None
            probe = probe[:-1]

    def __len__(self) -> int:
        """Number of recorded locations."""
        return len(self._locations)

    def merge_prefixed(self, other: SourceMap, prefix: SourcePath) -> None:
        """Merge another map's entries under *prefix* (include splicing)."""
        for path, loc in other._locations.items():
            self._locations[prefix + path] = loc
