"""Include preprocessing plugin for YAML file inclusion.

Resolves ``!include`` directives, supporting simple and anchor-based includes
with security protections against directory traversal and circular references.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from hexdag.compiler.preprocessing._type_guards import _is_dict_config
from hexdag.kernel.exceptions import YamlPipelineBuilderError
from hexdag.kernel.logging import get_logger

logger = get_logger(__name__)


class IncludePreprocessPlugin:
    """Resolve !include directives for YAML file inclusion.

    Supports two syntaxes:
    1. Simple include: !include path/to/file.yaml
    2. Anchor include: !include path/to/file.yaml#anchor_name

    Security:
    - Only allows relative paths (no absolute paths)
    - Prevents directory traversal attacks (no ../ beyond project root)
    - Detects circular includes

    For comprehensive examples, see notebooks/03_yaml_includes_and_composition.ipynb
    """

    def __init__(self, base_path: Path | None = None, max_depth: int = 10):
        """Initialize include plugin.

        Args:
            base_path: Base directory for relative includes (changeable via context manager)
            max_depth: Maximum include nesting depth to prevent circular includes
        """
        self.base_path = base_path or Path.cwd()
        self.project_root = self.base_path  # Fixed project root for security validation
        self.max_depth = max_depth

    def process(self, config: dict[str, Any]) -> dict[str, Any]:
        """Process !include directives recursively."""

        # Create new include stack for this processing run (thread-safe)
        include_stack: list[Path] = []
        result = self._resolve_includes(
            config, self.base_path, depth=0, include_stack=include_stack
        )
        if not _is_dict_config(result):
            raise TypeError(
                f"Include processing must return a dictionary, got {type(result).__name__}. "
                "Check that your included files resolve to valid YAML dictionaries."
            )
        return result

    def _resolve_includes(
        self, obj: Any, current_base: Path, depth: int, include_stack: list[Path]
    ) -> dict[str, Any] | list[Any] | Any:
        """Recursively resolve !include directives.

        Args:
            obj: Object to process (dict, list, or primitive)
            current_base: Base path for resolving relative includes
            depth: Current recursion depth
            include_stack: Stack of currently processing files (for circular detection)

        Returns:
            Processed object with includes resolved
        """

        if depth > self.max_depth:
            raise YamlPipelineBuilderError(
                f"Include nesting too deep (max {self.max_depth}). "
                f"Possible circular include in: {' -> '.join(str(p) for p in include_stack)}"
            )

        if isinstance(obj, dict):
            # Check for !include directive
            if "!include" in obj and len(obj) == 1:
                include_spec = obj["!include"]
                return self._load_include(include_spec, current_base, depth, include_stack)

            # Recurse into dict values
            return {
                k: self._resolve_includes(v, current_base, depth, include_stack)
                for k, v in obj.items()
            }

        if isinstance(obj, list):
            # Process each list item and flatten nested lists from includes
            result = []
            for item in obj:
                resolved = self._resolve_includes(item, current_base, depth, include_stack)
                # Flatten: if an include returns a list, extend rather than append
                if isinstance(resolved, list):
                    result.extend(resolved)
                else:
                    result.append(resolved)
            return result

        return obj

    def _load_include(
        self, include_spec: str, current_base: Path, depth: int, include_stack: list[Path]
    ) -> Any:
        """Load content from included file.

        Args:
            include_spec: Include specification (e.g., "file.yaml" or "file.yaml#anchor")
            current_base: Base path for resolving relative paths
            depth: Current recursion depth
            include_stack: Stack of currently processing files (for circular detection)

        Returns:
            Loaded and processed content from included file
        """

        # Parse include specification (strip whitespace for better UX)
        include_spec = include_spec.strip()
        if "#" in include_spec:
            file_path_str, anchor = include_spec.split("#", 1)
            file_path_str = file_path_str.strip()
            anchor = anchor.strip()
        else:
            file_path_str, anchor = include_spec, None

        # Resolve file path
        file_path = self._resolve_path(file_path_str, current_base)

        # Check for circular includes
        if file_path in include_stack:
            cycle = " -> ".join(str(p) for p in include_stack + [file_path])
            raise YamlPipelineBuilderError(f"Circular include detected: {cycle}")

        # Load YAML file
        try:
            include_stack.append(file_path)
            content = yaml.safe_load(file_path.read_text(encoding="utf-8"))

            # Extract anchor if specified
            if anchor:
                if not isinstance(content, dict) or anchor not in content:
                    raise YamlPipelineBuilderError(
                        f"Anchor '{anchor}' not found in {file_path}. "
                        f"Available: {list(content.keys()) if isinstance(content, dict) else 'N/A'}"
                    )
                content = content[anchor]

            # Recursively resolve includes in loaded content
            return self._resolve_includes(content, file_path.parent, depth + 1, include_stack)

        except FileNotFoundError as e:
            raise YamlPipelineBuilderError(
                f"Include file not found: {file_path}\nSearched relative to: {current_base}"
            ) from e
        except yaml.YAMLError as e:
            raise YamlPipelineBuilderError(f"Invalid YAML in included file {file_path}: {e}") from e
        finally:
            include_stack.pop()

    def _resolve_path(self, path_str: str, current_base: Path) -> Path:
        """Resolve and validate include path.

        Args:
            path_str: Path string from !include directive
            current_base: Base path for resolving relative paths

        Returns:
            Validated absolute path

        Raises:
            YamlPipelineBuilderError: If path is invalid or potentially malicious
        """

        # Prevent absolute paths
        if Path(path_str).is_absolute():
            raise YamlPipelineBuilderError(
                f"Absolute paths not allowed in !include: {path_str}\n"
                "Use relative paths only for security."
            )

        # Resolve path relative to current base
        resolved = (current_base / path_str).resolve()

        # Prevent directory traversal outside project root
        # Use the resolved project_root (not base_path) to handle symlinks properly
        resolved_root = self.project_root.resolve()
        try:
            resolved.relative_to(resolved_root)
        except ValueError as e:
            raise YamlPipelineBuilderError(
                f"Include path traverses outside project root: {path_str}\n"
                f"Project root: {resolved_root}\n"
                f"Attempted path: {resolved}"
            ) from e

        return resolved
