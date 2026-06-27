"""YAML Pipeline Validator - Validates pipeline configurations."""

import difflib
import inspect
import re
from collections import Counter
from collections.abc import Iterator
from typing import Any

from hexdag.compiler.reference_resolver import (
    _BUILTIN_NAMES,
    _INPUT_FIELD_RE,
    _NODE_FIELD_RE,
    extract_input_refs_from_mapping,
    extract_jinja_head_names,
    extract_refs_from_spec,
    iter_spec_strings,
)
from hexdag.kernel.context.execution_context import _NS_LOOKUP, RESERVED_NAMES
from hexdag.kernel.domain.dag import DirectedGraph
from hexdag.kernel.domain.pipeline_config import BaseNodeConfig
from hexdag.kernel.expression_parser import ALLOWED_FUNCTIONS
from hexdag.kernel.logging import get_logger
from hexdag.kernel.resolver import ResolveError, resolve

_logger = get_logger(__name__)

# Separator for namespace:name format
NAMESPACE_SEPARATOR = ":"

# Lazy-loaded known node types (derived from resolver's builtin aliases)
_known_node_types: frozenset[str] | None = None


def _get_known_node_types() -> frozenset[str]:
    """Lazily load known node types from resolver's builtin aliases.

    This derives valid node types from the aliases registered by hexdag.stdlib,
    maintaining hexagonal architecture (core doesn't import from builtin).
    Types are cached after first load.
    """
    global _known_node_types
    if _known_node_types is None:
        from hexdag.kernel.resolver import (
            get_builtin_aliases,  # lazy: avoid circular import with node discovery
        )

        _known_node_types = frozenset(get_builtin_aliases().keys())
    return _known_node_types


# Keep KNOWN_NODE_TYPES as a module-level reference for backwards compatibility
# in tests that may import it directly.
KNOWN_NODE_TYPES = _get_known_node_types()


class ValidationReport:
    """Container for validation results with optimized memory usage."""

    __slots__ = ("_errors", "_warnings", "_suggestions")

    def __init__(self) -> None:
        """Initialize validation result."""
        self._errors: list[str] = []
        self._warnings: list[str] = []
        self._suggestions: list[str] = []

    @property
    def is_valid(self) -> bool:
        """Check if validation passed (no errors).

        Returns
        -------
        bool
            True if no errors are present, False otherwise
        """
        return len(self._errors) == 0

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self._errors.append(message)

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self._warnings.append(message)

    def add_suggestion(self, message: str) -> None:
        """Add a suggestion message."""
        self._suggestions.append(message)

    @property
    def errors(self) -> list[str]:
        """Get all error messages.

        Returns
        -------
        list[str]
            List of error messages
        """
        return self._errors

    @property
    def warnings(self) -> list[str]:
        """Get all warning messages.

        Returns
        -------
        list[str]
            List of warning messages
        """
        return self._warnings

    @property
    def suggestions(self) -> list[str]:
        """Get all suggestion messages.

        Returns
        -------
        list[str]
            List of suggestion messages
        """
        return self._suggestions

    def __bool__(self) -> bool:
        """True if validation passed (no errors)."""
        return not self._errors

    def __len__(self) -> int:
        """Total number of issues (errors + warnings + suggestions)."""
        return len(self._errors) + len(self._warnings) + len(self._suggestions)

    def __iter__(self) -> Iterator[str]:
        """Iterate over all issues: errors first, then warnings, then suggestions."""
        yield from self._errors
        yield from self._warnings
        yield from self._suggestions


class _SchemaValidator:
    """Validates YAML node specs against known schemas.

    Since we no longer have a registry, schema validation is simplified
    to basic structural validation.

    Note: This is an internal class. Use YamlValidator for public validation API.
    """

    def validate_node_spec(
        self,
        node_type: str,
        spec: dict[str, Any],
        namespace: str = "core",
    ) -> list[str]:
        """Validate a node's spec with basic structural checks.

        Args
        ----
            node_type: Type of node (e.g., "llm", "agent", "function")
            spec: Node specification from YAML manifest
            namespace: Component namespace (default: "core")

        Returns
        -------
            List of validation error messages (empty if valid)
        """
        # Basic validation - without registry, we can only do structural checks
        errors: list[str] = []

        # LLM nodes require human_message (or legacy template/prompt_template)
        if (
            node_type in ("llm", "llm_node")
            and "human_message" not in spec
            and "template" not in spec
            and "prompt_template" not in spec
        ):
            errors.append("Missing required field 'human_message' (or legacy 'prompt_template')")

        # Prompt nodes require template
        if (
            node_type in ("prompt", "prompt_node")
            and "template" not in spec
            and "prompt_ref" not in spec
        ):
            errors.append("Missing required field 'template' or 'prompt_ref'")

        # Agent nodes require initial_prompt_template or main_prompt
        if (
            node_type in ("agent", "agent_node")
            and "initial_prompt_template" not in spec
            and "main_prompt" not in spec
        ):
            errors.append("Missing required field 'initial_prompt_template' (or 'main_prompt')")

        # Function nodes require fn
        if node_type in ("function", "function_node") and "fn" not in spec:
            errors.append("Missing required field 'fn'")

        # Composite nodes require mode
        if node_type in ("composite", "composite_node"):
            errors.extend(self._validate_composite_spec(spec))

        return errors

    def _validate_composite_spec(self, spec: dict[str, Any]) -> list[str]:
        """Validate composite_node specific requirements.

        Parameters
        ----------
        spec : dict[str, Any]
            Node specification from YAML manifest

        Returns
        -------
        list[str]
            List of validation error messages
        """
        errors: list[str] = []

        if "mode" not in spec:
            errors.append("Missing required field 'mode'")
            return errors

        mode = spec.get("mode")
        valid_modes = ("while", "for-each", "times", "if-else", "switch")
        if mode not in valid_modes:
            errors.append(f"Invalid mode '{mode}'. Valid modes: {', '.join(valid_modes)}")
            return errors

        # Mode-specific validation
        match mode:
            case "while":
                if "condition" not in spec:
                    errors.append("Mode 'while' requires 'condition' field")
            case "for-each":
                if "items" not in spec:
                    errors.append("Mode 'for-each' requires 'items' field")
            case "times":
                if "count" not in spec:
                    errors.append("Mode 'times' requires 'count' field")
                elif not isinstance(spec.get("count"), int):
                    errors.append("Field 'count' must be an integer")
            case "if-else":
                if "condition" not in spec:
                    errors.append("Mode 'if-else' requires 'condition' field")
            case "switch":
                if "branches" not in spec:
                    errors.append("Mode 'switch' requires 'branches' field")
                elif not isinstance(spec.get("branches"), list):
                    errors.append("Field 'branches' must be a list")

        # Validate body field if present (can be string, list, or callable from !py)
        body = spec.get("body")
        body_pipeline = spec.get("body_pipeline")

        if body is not None and body_pipeline is not None:
            errors.append("Cannot specify both 'body' and 'body_pipeline'")

        # Iterating modes require a body or body_pipeline
        if mode in ("while", "for-each", "times") and body is None and body_pipeline is None:
            errors.append(f"Mode '{mode}' requires 'body' or 'body_pipeline'")

        # Validate count is positive (0 is a silent no-op)
        if mode == "times" and isinstance(spec.get("count"), int) and spec["count"] <= 0:
            errors.append("Field 'count' must be a positive integer (> 0)")

        # Validate switch branch structure
        if mode == "switch" and isinstance(spec.get("branches"), list):
            for i, branch in enumerate(spec["branches"]):
                if not isinstance(branch, dict):
                    errors.append(f"branches[{i}] must be a dict")
                elif "condition" not in branch:
                    errors.append(f"branches[{i}] missing required 'condition' field")

        # If body is a callable (from !py tag), it's already validated at parse time
        # If body is a list, it should be inline nodes
        if isinstance(body, list):
            for i, node_config in enumerate(body):
                if not isinstance(node_config, dict):
                    errors.append(f"body[{i}] must be a node configuration dict")
                elif "kind" not in node_config:
                    errors.append(f"body[{i}] missing required 'kind' field")

        return errors


class YamlValidator:
    """Validates YAML pipeline configurations with optimized performance."""

    # Node kinds excluded from $input consistency checking — their input_mapping
    # serves a different purpose (expression aliases, not function params).
    _SKIP_CONSISTENCY_KINDS: frozenset[str] = frozenset({"expression_node", "expression"})

    def __init__(
        self,
        valid_node_types: set[str] | frozenset[str] | None = None,
        *,
        input_mapping_consistency_threshold: float = 0.5,
    ) -> None:
        """Initialize validator with configurable node types.

        Args
        ----
            valid_node_types: Set of valid node type names. If None, uses defaults.
            input_mapping_consistency_threshold: Fraction of sibling nodes that
                must reference a ``$input.X`` field before a missing reference
                becomes a build error (default 0.5 = majority).
        """
        self._provided_node_types = (
            frozenset(valid_node_types) if valid_node_types is not None else None
        )
        self._cached_node_types: frozenset[str] | None = None
        self._input_mapping_consistency_threshold = input_mapping_consistency_threshold

        # Schema validator for spec validation
        self.schema_validator = _SchemaValidator()

    @property
    def valid_node_types(self) -> frozenset[str]:
        """Get valid node types.

        Returns
        -------
        frozenset[str]
            Set of valid node type names
        """
        # If user provided explicit node types, use those
        if self._provided_node_types is not None:
            return self._provided_node_types

        # Otherwise, use auto-discovered node types
        if self._cached_node_types is None:
            self._cached_node_types = _get_known_node_types()

        return self._cached_node_types

    def validate(self, config: Any) -> ValidationReport:
        """Validate complete YAML configuration with optimized caching.

        Expects declarative manifest format: {kind: Pipeline,
        spec: {nodes: [{kind, metadata, spec}]}}

        Args
        ----
            config: Parsed YAML configuration

        Returns
        -------
        ValidationReport
            ValidationReport with errors, warnings, and suggestions
        """
        result = ValidationReport()

        # Validate manifest structure
        self._validate_manifest_structure(config, result)

        if not result:
            return result

        spec = config.get("spec", {})
        nodes = spec.get("nodes", [])
        pipeline_ports = spec.get("ports", {}) if isinstance(spec.get("ports"), dict) else {}

        # Validate nodes and cache the IDs and macro instances for reuse
        node_ids, macro_instances = self._validate_nodes(nodes, result, pipeline_ports)

        # Reuse cached node_ids and macro_instances for dependency validation
        dep_graph = self._validate_dependencies_with_cache(nodes, result, node_ids, macro_instances)

        # Validate expression/mapping naming to prevent ambiguous resolution
        self._validate_naming_collisions(nodes, result, node_ids, macro_instances)

        # Validate that all node references are declared as dependencies
        self._validate_undeclared_refs(nodes, dep_graph, node_ids, macro_instances, result)

        # Warn on {{name}} refs that look like typos of real node names
        self._validate_template_typos(nodes, dep_graph, node_ids, macro_instances, result)

        # Validate $input data-flow consistency using the dependency graph
        self._validate_input_flow_consistency(nodes, dep_graph, result)

        # Validate $input references against declared input_schema (if any)
        input_schema = spec.get("input_schema")
        if isinstance(input_schema, dict):
            self._validate_input_refs_against_schema(nodes, input_schema, result)

        return result

    def _validate_manifest_structure(self, config: Any, result: ValidationReport) -> None:
        """Validate declarative manifest YAML structure.

        Args
        ----
            config: Parsed YAML configuration
            result: ValidationReport to add errors to
        """
        if not isinstance(config, dict):
            result.add_error("Configuration must be a dictionary")
            return

        if "kind" not in config:
            result.add_error(
                "Configuration must contain 'kind' field (declarative manifest format required)"
            )
            return

        if "metadata" not in config:
            result.add_error("Configuration must contain 'metadata' field")
            return

        # Macro and Config definitions have different structure
        kind = config.get("kind")
        if kind == "Macro":
            # Macro has: metadata, parameters, nodes or nodes_raw (no spec)
            if "nodes" not in config and "nodes_raw" not in config:
                result.add_error("Macro definition must contain 'nodes' or 'nodes_raw' field")
                return
            # Skip rest of validation for Macro kind
            return

        if kind == "Config":
            # Config has: metadata, spec (no nodes)
            if "spec" not in config:
                result.add_error("Config definition must contain 'spec' field")
            # Skip pipeline-specific validation for Config kind
            return

        if kind == "Middleware":
            # Middleware has: metadata, spec.stack (no nodes)
            spec = config.get("spec", {})
            if not isinstance(spec, dict) or "stack" not in spec:
                result.add_error("Middleware definition must contain 'spec.stack' field")
            # Skip pipeline-specific validation for Middleware kind
            return

        if kind == "Adapter":
            # Adapter has: metadata, spec.class (no nodes)
            spec = config.get("spec", {})
            if not isinstance(spec, dict) or "class" not in spec:
                result.add_error("Adapter definition must contain 'spec.class' field")
            # Skip pipeline-specific validation for Adapter kind
            return

        # For Pipeline and other kinds, validate spec
        if "spec" not in config:
            result.add_error("Configuration must contain 'spec' field")
            return

        spec = config.get("spec", {})
        if not isinstance(spec, dict):
            result.add_error("'spec' field must be a dictionary")
            return

        if "nodes" not in spec:
            result.add_error("'spec' must contain 'nodes' field")
            return

        if not isinstance(spec["nodes"], list):
            result.add_error("'spec.nodes' field must be a list")
            return

        if len(spec["nodes"]) == 0:
            result.add_warning("Pipeline has no nodes defined")

        # Validate common_field_mappings structure if present
        common_mappings = spec.get("common_field_mappings")
        if common_mappings is not None and not isinstance(common_mappings, dict):
            result.add_error("'spec.common_field_mappings' must be a dictionary")

        # Rule 3 (bonus): Validate state machine structure
        state_machines = spec.get("state_machines")
        if isinstance(state_machines, dict):
            for sm_name, sm_spec in state_machines.items():
                if not isinstance(sm_spec, dict):
                    result.add_error(f"State machine '{sm_name}' must be a dict")
                    continue
                if "initial" not in sm_spec:
                    result.add_error(f"State machine '{sm_name}': missing required 'initial' field")
                transitions = sm_spec.get("transitions")
                if transitions is not None and not isinstance(transitions, dict):
                    result.add_error(f"State machine '{sm_name}': 'transitions' must be a dict")

    # Structural keys valid at the node level (alongside kind/metadata/spec).
    # ``dependencies`` and ``wait_for`` are read from node level by
    # ``BaseNodeConfig.from_node_config()``.
    # Fields that live inside spec, not at the node level
    _SPEC_ONLY_FIELDS: frozenset[str] = frozenset({"input_mapping", "strict_mapping", "when"})

    _VALID_NODE_LEVEL_KEYS: frozenset[str] = frozenset(
        {"kind", "metadata", "spec", "settings"}
        | {
            field_name
            for field_name in BaseNodeConfig.model_fields
            if field_name not in {"input_mapping", "strict_mapping", "when"}
        }
    )

    def _validate_nodes(
        self,
        nodes: list[dict[str, Any]],
        result: ValidationReport,
        pipeline_ports: dict[str, Any] | None = None,
    ) -> tuple[set[str], set[str]]:
        """Validate nodes and return node IDs and macro instance names.

        Expects declarative node format: {kind, metadata: {name}, spec: {dependencies}}

        Returns
        -------
        tuple[set[str], set[str]]
            Tuple of (node_ids, macro_instance_names) for caching and reuse in dependency validation
        """
        node_ids = set()
        macro_instances = set()

        for i, node in enumerate(nodes):
            # Validate node has required fields
            if "kind" not in node:
                result.add_error(f"Node {i}: Missing 'kind' field")
                continue

            if "metadata" not in node:
                result.add_error(f"Node {i}: Missing 'metadata' field")
                continue

            node_id = node.get("metadata", {}).get("name")
            if not node_id:
                result.add_error(f"Node {i}: Missing 'metadata.name'")
                continue

            # Detect fields misplaced at node level that belong inside spec
            for key in node:
                if key not in self._VALID_NODE_LEVEL_KEYS:
                    result.add_error(
                        f"Node '{node_id}': '{key}' is not a valid node-level field. "
                        f"Move it inside 'spec'."
                    )

            kind = node.get("kind", "")

            # Check node ID uniqueness
            if node_id in node_ids:
                result.add_error(f"Duplicate node ID: '{node_id}'")
            node_ids.add(node_id)

            # Special case: macro_invocation is not a node type, skip node type validation
            if kind == "macro_invocation":
                spec = node.get("spec", {})
                if "macro" not in spec:
                    result.add_error(
                        f"Node '{node_id}': macro_invocation must specify 'spec.macro' field"
                    )
                else:
                    # Rule 1: validate macro reference is resolvable
                    self._validate_resolvable(
                        spec["macro"],
                        node_id,
                        "macro",
                        result,
                    )
                macro_instances.add(node_id)
                # Validate when clause on macro invocations too
                macro_when = spec.get("config", {}).get("when") or spec.get("when")
                if macro_when and isinstance(macro_when, str):
                    self._validate_when_syntax(node_id, macro_when, result)
                continue

            # Merge settings + spec for parameter validation
            # settings: contains literal config, spec: contains dynamic wiring
            _settings = node.get("settings", {})
            _spec = node.get("spec", {})
            merged_params = {**_settings, **_spec}

            # Rule 1: Signature-based validation for ALL node kinds
            # (dotted paths, registered aliases, and builtin aliases).
            self._validate_spec_against_factory(
                kind,
                node_id,
                merged_params,
                pipeline_ports,
                result,
            )

            # Handle module paths (e.g., hexdag.stdlib.nodes.LLMNode)
            if "." in kind and ":" not in kind:
                # Full module path — signature validation above is sufficient;
                # skip legacy node-type-name checks below.
                continue

            # Handle user-registered aliases
            from hexdag.kernel.resolver import get_registered_aliases

            if kind in get_registered_aliases():
                continue

            if NAMESPACE_SEPARATOR in kind:
                namespace, node_kind = kind.split(NAMESPACE_SEPARATOR, 1)
            else:
                namespace = "core"
                node_kind = kind

            # Remove '_node' suffix if present
            node_type = node_kind[:-5] if node_kind.endswith("_node") else node_kind

            qualified_node_type = f"{namespace}:{node_type}"

            params = merged_params

            # Validate node type
            # Support both qualified (namespace:type) and simple (type) formats
            if (
                qualified_node_type not in self.valid_node_types
                and node_type not in self.valid_node_types
                and node_kind not in self.valid_node_types
            ):
                # Show available types grouped by namespace
                by_namespace: dict[str, list[str]] = {}
                simple_types: list[str] = []
                has_namespaced = False

                for valid_type in sorted(self.valid_node_types):
                    if ":" in valid_type:
                        has_namespaced = True
                        ns, nt = valid_type.split(":", 1)
                        by_namespace.setdefault(ns, []).append(nt)
                    else:
                        # Legacy format without namespace
                        simple_types.append(valid_type)

                parts = []
                if by_namespace:
                    parts.append(
                        ", ".join(
                            f"{ns}:[{', '.join(types)}]"
                            for ns, types in sorted(by_namespace.items())
                        )
                    )
                if simple_types:
                    parts.append(", ".join(sorted(simple_types)))

                valid_types_str = ", ".join(parts) if parts else "none"

                # Use simple node_type in error if no valid types have namespaces (legacy mode)
                invalid_type_str = node_type if not has_namespaced else qualified_node_type

                # Suggest close matches for likely typos
                close = difflib.get_close_matches(
                    node_kind, sorted(self.valid_node_types), n=3, cutoff=0.5
                )
                hint = f" Did you mean: {', '.join(close)}?" if close else ""

                result.add_error(
                    f"Node '{node_id}': Invalid type '{invalid_type_str}'. "
                    f"Valid types: {valid_types_str}.{hint}"
                )

            # Validate node-specific requirements and schema
            self._validate_node_params(node_id, node_type, params, namespace, result)

            # Validate when clause syntax (if present)
            when_clause = _spec.get("when") or merged_params.get("when")
            if when_clause and isinstance(when_clause, str):
                self._validate_when_syntax(node_id, when_clause, result)

            # Validate strict_mapping requires input_mapping
            if _spec.get("strict_mapping") and not _spec.get("input_mapping"):
                result.add_error(
                    f"Node '{node_id}': strict_mapping requires input_mapping to be set"
                )

        return node_ids, macro_instances

    @staticmethod
    def _validate_when_syntax(
        node_id: str | None,
        when_clause: str,
        result: ValidationReport,
    ) -> None:
        """Validate that a ``when`` clause is syntactically valid.

        Calls ``compile_expression`` at build time to catch syntax errors
        early instead of failing at runtime.
        """
        from hexdag.kernel.expression_parser import (
            compile_expression,  # noqa: PLC0415  # lazy: avoid circular import
        )

        try:
            compile_expression(when_clause)
        except Exception as exc:  # noqa: BLE001
            result.add_error(f"Node '{node_id}': invalid 'when' expression: {exc}")

    def _validate_node_params(
        self,
        node_id: str | None,
        node_type: str,
        params: dict[str, Any],
        namespace: str,
        result: ValidationReport,
    ) -> None:
        """Validate node-specific parameters using basic structural validation.

        Args
        ----
            node_id: Node identifier
            node_type: Type of node (e.g., "llm", "function")
            params: Node spec parameters
            namespace: Component namespace
            result: ValidationReport to add errors to
        """
        # Schema-based validation
        schema_errors = self.schema_validator.validate_node_spec(
            node_type, params, namespace=namespace
        )
        for error in schema_errors:
            result.add_error(f"Node '{node_id}': {error}")

    def _validate_dependencies_with_cache(
        self,
        nodes: list[dict[str, Any]],
        result: ValidationReport,
        node_ids: set[str],
        macro_instances: set[str],
    ) -> dict[str, set[str]]:
        """Validate node dependencies using cached node IDs and check for cycles.

        Dependencies are in spec.dependencies field.

        Parameters
        ----------
        nodes : list[dict[str, Any]]
            List of node configurations
        result : ValidationReport
            Report to add errors to
        node_ids : set[str]
            Cached set of valid node IDs from _validate_nodes
        macro_instances : set[str]
            Set of macro instance names (nodes will be generated at runtime)

        Returns
        -------
        dict[str, set[str]]
            Dependency graph ``{node_id: set_of_dependency_ids}`` for reuse
            by downstream validation rules.
        """
        dependency_graph: dict[str, set[str]] = {}

        for node in nodes:
            node_id = node.get("metadata", {}).get("name")
            if not node_id:
                continue

            # Parse base fields via the model — single source of truth
            base = BaseNodeConfig.from_node_config(node)
            all_deps = list(base.dependencies or []) + list(base.wait_for or [])

            # Check all dependencies exist using cached node_ids
            valid_deps = set()
            for dep in all_deps:
                if dep in node_ids:
                    valid_deps.add(dep)
                    continue

                is_macro_generated = False
                for macro_instance in macro_instances:
                    if dep.startswith(f"{macro_instance}_"):
                        is_macro_generated = True
                        valid_deps.add(dep)
                        break

                # If not a known node and not macro-generated, it's an error
                if not is_macro_generated:
                    result.add_error(f"Node '{node_id}': Dependency '{dep}' does not exist")

            dependency_graph[node_id] = valid_deps

        # Validate on_error references
        for node in nodes:
            node_id = node.get("metadata", {}).get("name")
            if not node_id:
                continue
            on_error = node.get("spec", {}).get("on_error")
            if on_error is not None:
                if on_error == node_id:
                    result.add_error(f"Node '{node_id}': on_error cannot reference itself")
                elif on_error not in node_ids and not any(
                    on_error.startswith(f"{mi}_") for mi in macro_instances
                ):
                    result.add_error(
                        f"Node '{node_id}': on_error handler '{on_error}' does not exist"
                    )

        # Check for cycles using DirectedGraph's public static method
        if cycle_message := DirectedGraph.detect_cycle(dependency_graph):
            result.add_error(cycle_message)

        return dependency_graph

    def _validate_naming_collisions(
        self,
        nodes: list[dict[str, Any]],
        result: ValidationReport,
        node_ids: set[str],
        macro_instances: set[str],
    ) -> None:
        """Validate that expression variables and input_mapping aliases don't collide.

        Checks (all hard errors):
        1. Expression variable name == node name
        2. Expression variable name == builtin function name
        3. input_mapping alias == node name
        4. First path segment in input_mapping/expressions is a known reference
        """
        builtin_names = frozenset(ALLOWED_FUNCTIONS.keys())

        for node in nodes:
            node_id = node.get("metadata", {}).get("name")
            if not node_id:
                continue

            spec = node.get("spec", {})
            expressions = spec.get("expressions", {})
            input_mapping = spec.get("input_mapping", {})

            if not isinstance(expressions, dict):
                expressions = {}
            if not isinstance(input_mapping, dict):
                input_mapping = {}

            # Check 1: Expression variable names vs node names
            for var_name in expressions:
                if var_name in node_ids:
                    result.add_error(
                        f"Node '{node_id}': Expression variable '{var_name}' "
                        f"collides with node '{var_name}'. Rename the variable "
                        f"or the node to avoid ambiguous resolution."
                    )

            # Check 2: Expression variable names vs builtin functions
            for var_name in expressions:
                if var_name in builtin_names:
                    result.add_error(
                        f"Node '{node_id}': Expression variable '{var_name}' "
                        f"collides with built-in function '{var_name}'. "
                        f"Choose a different variable name."
                    )

            # Check 3: input_mapping alias vs node names
            for alias in input_mapping:
                if alias in node_ids:
                    result.add_error(
                        f"Node '{node_id}': input_mapping alias '{alias}' "
                        f"collides with node '{alias}'. Choose a different alias."
                    )

            # Check 3b: Expression variable names vs input_mapping aliases
            overlap = set(expressions.keys()) & set(input_mapping.keys())
            if overlap:
                for name in sorted(overlap):
                    result.add_error(
                        f"Node '{node_id}': expression variable '{name}' "
                        f"collides with input_mapping alias '{name}'. "
                        f"Rename one to avoid ambiguous resolution."
                    )

            # Check 4: First path segment validation in expressions
            # Build the set of valid first segments for this node
            valid_first_segments = (
                node_ids
                | set(expressions.keys())
                | set(input_mapping.keys())
                | RESERVED_NAMES
                | _BUILTIN_NAMES
            )

            for var_name, expr in expressions.items():
                if not isinstance(expr, str):
                    continue
                self._check_first_segments(
                    expr, var_name, node_id, valid_first_segments, node_ids, macro_instances, result
                )

            for alias, source in input_mapping.items():
                if not isinstance(source, str):
                    continue
                # Validate expression syntax for mapping values that
                # contain operators or function calls (expressions, not simple refs)
                if any(
                    op in source
                    for op in (
                        "==",
                        "!=",
                        "<=",
                        ">=",
                        " < ",
                        " > ",
                        " + ",
                        " - ",
                        " * ",
                        " / ",
                        " % ",
                        " and ",
                        " or ",
                        " not ",
                        " if ",
                        " else ",
                        " in ",
                    )
                ) or any(re.search(rf"\b{re.escape(fn)}\(", source) for fn in ALLOWED_FUNCTIONS):
                    self._validate_when_syntax(node_id, source, result)
                # Skip $input references and expressions
                if source.startswith("$input") or source == "$input":
                    continue
                self._check_first_segments(
                    source, alias, node_id, valid_first_segments, node_ids, macro_instances, result
                )
                # Also validate bare names (no dot) — _check_first_segments only
                # catches dotted paths via regex; a bare input_mapping value like
                # "typo_node" would be silently ignored.
                stripped = source.strip()
                if (
                    "." not in stripped
                    and stripped.isidentifier()
                    and stripped not in _BUILTIN_NAMES
                    and stripped not in valid_first_segments
                    and not any(stripped.startswith(f"{mi}_") for mi in macro_instances)
                ):
                    close_matches = difflib.get_close_matches(
                        stripped, sorted(node_ids), n=3, cutoff=0.6
                    )
                    suggestion = ""
                    if close_matches:
                        suggestion = f" Did you mean: {', '.join(close_matches)}?"
                    result.add_warning(
                        f"Node '{node_id}': input_mapping alias '{alias}' "
                        f"references bare name '{stripped}' which is not a known node "
                        f"or expression variable.{suggestion}"
                    )

            # Rule 2: Validate references in composite/conditional fields.
            # These are the same fields yaml_builder._infer_deps() scans,
            # so typos here would cause runtime ExpressionError.
            for field_key in ("condition", "items", "when"):
                field_val = spec.get(field_key)
                if isinstance(field_val, str):
                    self._check_first_segments(
                        field_val,
                        field_key,
                        node_id,
                        valid_first_segments,
                        node_ids,
                        macro_instances,
                        result,
                    )

            branches = spec.get("branches")
            if isinstance(branches, list):
                for i, branch in enumerate(branches):
                    if isinstance(branch, dict):
                        cond = branch.get("condition")
                        if isinstance(cond, str):
                            self._check_first_segments(
                                cond,
                                f"branches[{i}].condition",
                                node_id,
                                valid_first_segments,
                                node_ids,
                                macro_instances,
                                result,
                            )

            state_update = spec.get("state_update")
            if isinstance(state_update, dict):
                for var_name, expr in state_update.items():
                    if isinstance(expr, str):
                        self._check_first_segments(
                            expr,
                            f"state_update.{var_name}",
                            node_id,
                            valid_first_segments,
                            node_ids,
                            macro_instances,
                            result,
                        )

    # ==================================================================
    # Rule 6: Undeclared node references
    # ==================================================================

    def _validate_undeclared_refs(
        self,
        nodes: list[dict[str, Any]],
        dep_graph: dict[str, set[str]],
        node_ids: set[str],
        macro_instances: set[str],
        result: ValidationReport,
    ) -> None:
        """Warn when a node references another node that is not in its
        explicit ``dependencies`` list.

        Uses the same :func:`extract_refs_from_spec` scan as the builder's
        ``_infer_deps``, so builder and validator can never disagree about
        what counts as a reference. The builder auto-merges these missing
        deps at build time (emitting the same warning), so the pipeline
        executes correctly either way — an incomplete explicit list is
        declaration hygiene, not a correctness error. Explicit deps are a
        supplement to inference, not an exhaustive promise.
        """
        known = frozenset(node_ids)

        for node in nodes:
            node_id = node.get("metadata", {}).get("name")
            if not node_id:
                continue

            # Only check nodes that have explicit dependencies declared —
            # nodes without explicit deps rely entirely on auto-inference.
            has_explicit_deps = "dependencies" in node or "dependencies" in node.get("spec", {})
            if not has_explicit_deps:
                continue

            declared_deps = dep_graph.get(node_id, set())
            other_nodes = known - {node_id}
            mi = frozenset(macro_instances)

            # Same scan the builder runs — single source of truth.
            inferred = extract_refs_from_spec(node.get("spec", {}), other_nodes, mi)

            missing = inferred - declared_deps
            # Exclude macro-generated nodes (they resolve at macro expansion)
            missing = {
                ref for ref in missing if not any(ref.startswith(f"{m}_") for m in macro_instances)
            }

            for ref in sorted(missing):
                result.add_warning(
                    f"Node '{node_id}': references node '{ref}' "
                    f"but '{ref}' is not in its explicit dependencies. "
                    f"The builder will add it automatically. "
                    f"Add '{ref}' to dependencies or remove the explicit "
                    f"dependencies key to let the builder infer them."
                )

    def _validate_template_typos(
        self,
        nodes: list[dict[str, Any]],
        dep_graph: dict[str, set[str]],
        node_ids: set[str],
        macro_instances: set[str],
        result: ValidationReport,
    ) -> None:
        """Warn on ``{{name}}`` refs whose name matches no node but is a
        close match of one — almost always a typo.

        A typo'd reference creates no dependency edge and resolves to
        ``None`` at runtime, so it fails silently. Two precision guards
        keep noise down:

        - Only close matches of real node names warn — template vars
          legitimately come from input_mapping aliases, ``$input``
          fields, and dependency output fields.
        - If the suggested node is already upstream (declared dep,
          inferred ref, or the implicit-chain predecessor), the name is
          most plausibly one of its *output fields* — single-dep nodes
          receive that output flat, so ``{{thread_context}}`` fed by
          ``fetch_thread_context`` resolves fine. A genuine typo breaks
          inference, so the suggested node is typically not upstream.
        """
        known = frozenset(node_ids)
        node_id_list = sorted(node_ids)
        mi = frozenset(macro_instances)
        previous_node_id: str | None = None

        for node in nodes:
            node_id = node.get("metadata", {}).get("name")
            if not node_id:
                continue
            spec = node.get("spec", {})

            # Names that legitimately appear in templates without being nodes
            allowed = (
                known
                | _BUILTIN_NAMES
                | RESERVED_NAMES
                # loop-scope variables available inside composite bodies
                | {"item", "index", "loop"}
            )
            input_mapping = spec.get("input_mapping")
            if isinstance(input_mapping, dict):
                allowed |= set(input_mapping.keys())
            expressions = spec.get("expressions")
            if isinstance(expressions, dict):
                allowed |= set(expressions.keys())

            candidates: set[str] = set()
            for text in iter_spec_strings(spec):
                if "{{" in text:
                    candidates |= extract_jinja_head_names(text)

            unknown = sorted(candidates - allowed)
            if unknown:
                upstream = dep_graph.get(node_id, set()) | extract_refs_from_spec(
                    spec, known - {node_id}, mi
                )
                if previous_node_id:
                    upstream.add(previous_node_id)

                for name in unknown:
                    if any(name.startswith(f"{m}_") for m in macro_instances):
                        continue
                    close = difflib.get_close_matches(name, node_id_list, n=1, cutoff=0.8)
                    if close and close[0] not in upstream:
                        result.add_warning(
                            f"Node '{node_id}': template references "
                            f"'{{{{{name}}}}}' but no node named '{name}' exists. "
                            f"Did you mean '{close[0]}'? Unknown references "
                            f"resolve to None at runtime."
                        )

            previous_node_id = node_id

    # ==================================================================
    # Rule 1: Signature-based spec validation
    # ==================================================================

    # Params that belong to the framework, not to the user's YAML spec.
    _SKIP_PARAMS: frozenset[str] = frozenset({
        "self",
        "cls",
        "name",
        "deps",
        "dependencies",
        "args",
        "kwargs",
    })

    def _validate_resolvable(
        self,
        module_path: str,
        node_id: str,
        label: str,
        result: ValidationReport,
    ) -> None:
        """Check that *module_path* can be resolved to an importable object.

        Reports a warning (not error) because macros may be registered
        dynamically via YAML macro definitions in multi-document files.
        """
        try:
            resolve(module_path)
        except (ResolveError, Exception):
            if "." in module_path:
                result.add_warning(
                    f"Node '{node_id}': Cannot resolve {label} '{module_path}'. "
                    f"Check the module path."
                )

    def _validate_spec_against_factory(
        self,
        kind: str,
        node_id: str,
        merged_spec: dict[str, Any],
        pipeline_ports: dict[str, Any] | None,
        result: ValidationReport,
    ) -> None:
        """Validate spec against factory __call__ signature and port requirements.

        Generic rule: resolve the factory class, introspect its ``__call__``
        parameters, and verify that every *required* parameter (no default)
        is present in the merged spec dict.  Also checks port requirements.
        """
        try:
            factory_obj = resolve(kind)
        except (ResolveError, Exception):
            # Can't resolve — may be a custom type registered at runtime.
            # Only warn for dotted paths (explicit module refs that should be importable).
            if "." in kind:
                result.add_error(
                    f"Node '{node_id}': Cannot resolve kind '{kind}'. Check the module path."
                )
            return

        # Get callable to introspect
        target = factory_obj
        if isinstance(factory_obj, type):  # pyright: ignore[reportUnnecessaryIsInstance]
            try:
                target = factory_obj()
            except Exception:
                return  # factory requires init args — skip introspection

        try:
            sig = inspect.signature(target)
        except (ValueError, TypeError):
            return  # can't introspect — skip gracefully

        for param_name, param in sig.parameters.items():
            if param_name in self._SKIP_PARAMS:
                continue
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            # Required = no default value
            if param.default is inspect.Parameter.empty and param_name not in merged_spec:
                result.add_error(
                    f"Node '{node_id}': Missing required field '{param_name}' for kind '{kind}'."
                )

        # Rule 3: Port requirement validation
        # Ports may be configured at runtime via port_overrides, so
        # missing ports are warnings, not errors.
        factory_cls = factory_obj if isinstance(factory_obj, type) else type(factory_obj)  # noqa: E501 # pyright: ignore[reportUnnecessaryIsInstance]
        port_caps = getattr(factory_cls, "_hexdag_port_capabilities", None)
        if port_caps and pipeline_ports is not None:
            for port_name in port_caps:
                if port_name not in pipeline_ports:
                    result.add_warning(
                        f"Node '{node_id}': Kind '{kind}' requires port "
                        f"'{port_name}' but it is not declared in spec.ports."
                    )

    @staticmethod
    def _check_first_segments(
        text: str,
        field_name: str,
        node_id: str,
        valid_first_segments: set[str],
        node_ids: set[str],
        macro_instances: set[str],
        result: "ValidationReport",
    ) -> None:
        """Check that first path segments in text are known references."""
        for match in _NODE_FIELD_RE.finditer(text):
            candidate = match.group(1)
            if candidate in _BUILTIN_NAMES:
                continue

            # Expression namespace — validate field if schema is known
            ns = _NS_LOOKUP.get(candidate)
            if ns is not None:
                if ns.fields is not None:
                    parts = match.group(0).split(".")
                    if len(parts) >= 2 and parts[1] not in ns.fields:
                        close = difflib.get_close_matches(
                            parts[1], sorted(ns.fields), n=2, cutoff=0.5
                        )
                        hint = f" Did you mean: {', '.join(close)}?" if close else ""
                        result.add_error(
                            f"Node '{node_id}': Unknown {ns.name} field '{parts[1]}' "
                            f"in '{field_name}'. "
                            f"Valid fields: {', '.join(sorted(ns.fields))}.{hint}"
                        )
                continue

            if candidate in valid_first_segments:
                continue
            # Check if candidate is a macro-expanded node name
            if any(candidate.startswith(f"{mi}_") for mi in macro_instances):
                continue
            # Unknown first segment — likely a typo
            close_matches = difflib.get_close_matches(candidate, sorted(node_ids), n=3, cutoff=0.6)
            suggestion = ""
            if close_matches:
                suggestion = f" Did you mean: {', '.join(close_matches)}?"
            result.add_error(
                f"Node '{node_id}': Unknown reference '{candidate}' "
                f"in field '{field_name}'.{suggestion}"
            )

    # ==================================================================
    # Rule 4: Graph-based $input data-flow consistency
    # ==================================================================

    def _validate_input_flow_consistency(
        self,
        nodes: list[dict[str, Any]],
        dep_graph: dict[str, set[str]],
        result: ValidationReport,
    ) -> None:
        """Error when sibling nodes on the same DAG branch inconsistently
        pass ``$input.X`` fields.

        Uses the dependency graph to find **graph siblings** — nodes that
        share at least one common parent.  Within each sibling group, if
        a ``$input.X`` field is used by ≥ threshold fraction of siblings
        but one omits it, that node gets a build error.

        This catches copy-paste omissions like ``send_counter_and_mc_request``
        missing ``conversation_id: $input.conversation_id`` while its four
        graph-siblings all pass it.

        Nodes can suppress the check for specific fields via
        ``validated_input_fields: [field_name]`` in their spec.
        """
        # Build a lookup: node_id → node config
        node_by_id: dict[str, dict[str, Any]] = {}
        for node in nodes:
            nid = node.get("metadata", {}).get("name", "")
            if nid:
                node_by_id[nid] = node

        # Build forward graph (parent → children)
        forward: dict[str, set[str]] = {}
        for child, parents in dep_graph.items():
            for parent in parents:
                forward.setdefault(parent, set()).add(child)

        # For each node, extract $input refs and validated_input_fields
        node_input_refs: dict[str, set[str]] = {}
        node_validated: dict[str, set[str]] = {}
        for nid, node in node_by_id.items():
            kind = node.get("kind", "")
            bare_kind = kind[:-5] if kind.endswith("_node") else kind
            if kind in self._SKIP_CONSISTENCY_KINDS or bare_kind in self._SKIP_CONSISTENCY_KINDS:
                continue
            spec = node.get("spec", {})
            input_mapping = spec.get("input_mapping")
            if not isinstance(input_mapping, dict) or not input_mapping:
                continue
            node_input_refs[nid] = extract_input_refs_from_mapping(input_mapping)
            node_validated[nid] = set(spec.get("validated_input_fields") or [])

        # Find sibling groups: nodes that share a common parent
        # A node can appear in multiple sibling groups (if it has multiple parents)
        sibling_groups: dict[str, set[str]] = {}
        for parent, children in forward.items():
            # Filter to children that have input_mapping (are in node_input_refs)
            eligible = children & node_input_refs.keys()
            if len(eligible) >= 2:
                sibling_groups[parent] = eligible

        # Also treat root nodes (no dependencies) as a sibling group
        root_nodes = {nid for nid in node_input_refs if not dep_graph.get(nid)}
        if len(root_nodes) >= 2:
            sibling_groups["__root__"] = root_nodes

        # Check consistency within each sibling group
        threshold = self._input_mapping_consistency_threshold
        reported: set[tuple[str, str]] = set()  # avoid duplicate errors

        for siblings in sibling_groups.values():
            field_usage: Counter[str] = Counter()
            for nid in siblings:
                field_usage.update(node_input_refs[nid])

            total = len(siblings)
            for nid in siblings:
                referenced = node_input_refs[nid]
                validated = node_validated.get(nid, set())
                for field, count in field_usage.items():
                    if (
                        count / total >= threshold
                        and field not in referenced
                        and field not in validated
                        and (nid, field) not in reported
                    ):
                        reported.add((nid, field))
                        result.add_error(
                            f"Node '{nid}': input_mapping is missing '$input.{field}' "
                            f"({count}/{total} sibling nodes pass it). "
                            f"Add '{field}: $input.{field}' to input_mapping, "
                            f"or add 'validated_input_fields: [{field}]' to suppress."
                        )

    # ==================================================================
    # Rule 5: $input references vs declared input_schema
    # ==================================================================

    def _validate_input_refs_against_schema(
        self,
        nodes: list[dict[str, Any]],
        input_schema: dict[str, Any],
        result: ValidationReport,
    ) -> None:
        """Error when a node references ``$input.X`` but *X* is not declared
        in the pipeline's ``input_schema``.

        Only runs when ``spec.input_schema`` is present (it is optional).
        """
        declared_fields = frozenset(input_schema.keys())

        for node in nodes:
            node_id = node.get("metadata", {}).get("name", "")
            if not node_id:
                continue
            spec = node.get("spec", {})

            # Collect $input refs from input_mapping
            unknown: set[str] = set()
            input_mapping = spec.get("input_mapping")
            if isinstance(input_mapping, dict):
                unknown |= extract_input_refs_from_mapping(input_mapping) - declared_fields

            # Collect $input refs from expressions
            expressions = spec.get("expressions")
            if isinstance(expressions, dict):
                for expr in expressions.values():
                    if isinstance(expr, str):
                        for match in _INPUT_FIELD_RE.finditer(expr):
                            field = match.group(1)
                            if field not in declared_fields:
                                unknown.add(field)

            # Collect $input refs from templates
            for tmpl_key in ("prompt_template", "template", "human_message", "system_message"):
                tmpl = spec.get(tmpl_key)
                if isinstance(tmpl, str):
                    for match in _INPUT_FIELD_RE.finditer(tmpl):
                        field = match.group(1)
                        if field not in declared_fields:
                            unknown.add(field)

            for field in sorted(unknown):
                result.add_error(
                    f"Node '{node_id}': references '$input.{field}' but pipeline "
                    f"input_schema does not declare field '{field}'. "
                    f"Declared fields: {', '.join(sorted(declared_fields))}."
                )
